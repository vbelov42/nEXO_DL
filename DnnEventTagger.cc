#include "DnnEventTagger.hh"

#include "SniperKernel/AlgFactory.h"
#include "SniperKernel/AlgBase.h"
#include "SniperKernel/SniperPtr.h"
#include "SniperKernel/SniperDataPtr.h"
#include "BufferMemMgr/EvtDataPtr.h"
#include "Event/EventObject.h"
#include "BufferMemMgr/IDataMemMgr.h"
#include "Event/SimHeader.h"
//#include "Event/ElecHeader.h"
#include "Event/PidTmvaHeader.h"

#include <algorithm>
#include <boost/python/numpy.hpp>
namespace py = boost::python;
namespace np = boost::python::numpy;

DECLARE_ALGORITHM(DnnEventTagger);

DnnEventTagger::DnnEventTagger(const std::string& name)
  : AlgBase(name)
{
  declProp("Pitch",     fPitch=3); //
}

DnnEventTagger::~DnnEventTagger()
{

}

bool DnnEventTagger::initialize()
{
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  Py_Initialize(); // is it safe?
  np::initialize();
  py::object module = py::import("DnnEventTagger").attr("DnnHelper");
  py::object init = module.attr("init_tagger");
  char filename[256]; snprintf(filename,sizeof(filename),"./checkpoint_%dmm_cl/ckpt.t7",fPitch);
  auto res = py::call<bool>(py::object(init).ptr(), filename);
  return bool(res);
}

bool DnnEventTagger::execute()
{
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  EvtDataPtr<nEXO::SimHeader> simheader(getParent(), "/Event/Sim");
  if (simheader.invalid()) {
    LogError << "can't find the simulation header." << std::endl;
    return false;
  }
  nEXO::SimEvent* simevent = dynamic_cast<nEXO::SimEvent*>(simheader->event());
  double fGenZ = simevent->GenZ();
  std::cout << "event irun="<< simheader->RunID() <<" iev="<< simheader->EventID() <<" type=" << simheader->getEventType() <<" "<< simevent << std::endl;
  std::cout << "      pos=("<< simevent->GenX() <<","<< simevent->GenY() <<","<< simevent->GenZ() <<") type=" << "simevent->GenParticleID()" <<" E="<< simevent->GenKineticE() << std::endl;

  EvtDataPtr<nEXO::PidTmvaHeader> tmvaheader(getParent(), "/Event/PidTmva");
  if (tmvaheader.invalid()) {
    LogError << "can't find the digitization header." << std::endl;
    return false;
  }
  nEXO::PidTmvaHeader* tmvaevent = tmvaheader.data();
  //nEXO::PidTmvaEvent* tmvaevent = dynamic_cast<nEXO::PidTmvaEvent*>(tmvaheader->event());
  std::cout << "      Etot=" << tmvaevent->get_TotalEnergy() <<" SD="<< tmvaevent->get_edge_distance() <<" T="<< tmvaevent->get_mean_time() <<" dT="<< tmvaevent->get_drift_time() << std::endl;

  const auto& xposition = tmvaevent->GetXPosition(), yposition = tmvaevent->GetYPosition();
  const auto& xcharge = tmvaevent->GetXCharge(), ycharge = tmvaevent->GetYCharge();
  const auto& xwf = tmvaevent->GetXWf(), ywf = tmvaevent->GetYWf();

  // skip big events
  double avedx = 0., avedy = 0.;
  if (!xposition.empty()) {
    double xtotcharge = 0., xcenter = 0.;
    for(unsigned i = 0; i < xposition.size(); i++) {
      xcenter += xposition[i]*xcharge[i];
      //txcenter += xtime[i]*xcharge[i];
      xtotcharge += xcharge[i];
    }    
    if (xtotcharge > 0.) {
      xcenter /= xtotcharge;
      //txcenter /= xtotcharge;
    }
    for(unsigned i = 0; i < xposition.size(); i++) {
      avedx += std::abs(xposition[i] - xcenter)*xcharge[i];
      //dtx += std::abs(xtime[i] - txcenter)*xcharge[i];
    }
    if (xtotcharge > 0.) {
      avedx /= xtotcharge;
      //dtx /= xtotcharge;
    }
  }
  if (!yposition.empty()) {
    double ytotcharge = 0., ycenter = 0.;
    for(unsigned i = 0; i < yposition.size(); i++) {
      ycenter += yposition[i]*ycharge[i];
      //tycenter += ytime[i]*ycharge[i];
      ytotcharge += ycharge[i];
    }    
    if (ytotcharge > 0.) {
      ycenter /= ytotcharge;
      //tycenter /= ytotcharge;
    }
    for(unsigned i = 0; i < yposition.size(); i++) {
      avedy += std::abs(yposition[i] - ycenter)*ycharge[i];
      //dty += std::abs(ytime[i] - tycenter)*ycharge[i];
    }
    if (ytotcharge > 0.) {
      avedy /= ytotcharge;
      //dty /= ytotcharge;
    }
  }
  // NB: E(abs(x-E(x))) != D(x) => avedx isn't dispersion
  const double max_size = 25.;
  if (hypot(avedx,avedy) > max_size)
    return true;

  // skip big events
  double xmin = 0., xmax = 0., ymin = 0., ymax = 0.;
  if (!xposition.empty()) {
    xmin = *std::min_element(xposition.begin(),xposition.end()), xmax = *std::max_element(xposition.begin(), xposition.end());
  }
  if (!yposition.empty()) {
    ymin = *std::min_element(yposition.begin(),yposition.end()), ymax = *std::max_element(yposition.begin(), yposition.end());
  }
  const double max_span = fPitch*180; // why?
  if (xmax - xmin > max_span || ymax - ymin > max_span)
    return true;

  // generate data
  const unsigned c_size = 200;
  const unsigned t_size = 255;
  double escale = 91100./tmvaevent->get_charge();
  np::ndarray img = np::zeros(py::make_tuple(c_size, t_size, 3), np::dtype::get_builtin<float>());
  for (unsigned i = 0; i < xposition.size(); i++) {
    int h = int((xposition[i] - xmin)/fPitch)+10;
    int wflen = xwf[i].size()-1;
    int samplet = 0;
    for (unsigned j = 0; j < t_size; j++) {
      if      (j< 80) samplet =        2*(j);
      else if (j<150) samplet = 160 +  4*(j- 80);
      else if (j<200) samplet = 440 +  6*(j-150);
      else            samplet = 740 + 16*(j-200);
      int k = h*img.strides(0)+j*img.strides(1)+0*img.strides(2);
      if (samplet < wflen)
        *(float*)(img.get_data()+k) += (xwf[i][wflen-samplet]/40.*escale + 25);
    }
  }
  for (unsigned i = 0; i < yposition.size(); i++) {
    int h = int((yposition[i] - ymin)/fPitch)+10;
    int wflen = ywf[i].size()-1;
    int samplet = 0;
    for (unsigned j = 0; j < t_size; j++) {
      if      (j< 80) samplet =        2*(j);
      else if (j<150) samplet = 160 +  4*(j- 80);
      else if (j<200) samplet = 440 +  6*(j-150);
      else            samplet = 740 + 16*(j-200);
      int k = h*img.strides(0)+j*img.strides(1)+1*img.strides(2);
      if (samplet < wflen)
        *(float*)(img.get_data()+k) += (ywf[i][wflen-samplet]/40.*escale + 25);
    }
  }
  std::cout << "    done image" << std::endl;

  // now send it to python
  //py::object main = py::import("__main__");
  //py::object module = py::import("DnnEventTagger");
  py::object helper = py::import("DnnEventTagger").attr("DnnHelper");
  if (1) {
    py::object save = helper.attr("save_image");
    //PyObject* func = py::object(main.attr("save_image")).ptr();
    char filename[256]; snprintf(filename, sizeof(filename), "images_%d/run_%04d-event_%04d.png", fPitch, simheader->RunID(), simheader->EventID());
    py::call<void>(py::object(save).ptr(), img, filename);
  }
  if (1) {
    py::object tag  = helper.attr("tag_event");
    auto res = py::call<double>(py::object(tag).ptr(), img);
    std::cout <<"call "<< py::extract<char const *>(py::str(tag)) <<" "<< res << std::endl;
  }
  
  return true;
}

bool DnnEventTagger::finalize()
{
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return true;
}

