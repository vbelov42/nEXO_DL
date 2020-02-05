#ifndef DnnEventTagger_hh
#define DnnEventTagger_hh

#include "SniperKernel/AlgBase.h"
#include <string>
// stl_loader.h
#include <vector>

#ifdef __CINT__
#pragma link C++ class std::vector<std::vector<double> >;
#else
template class std::vector<std::vector<double> >;
#endif


class DnnEventTagger: public AlgBase
{
public:
  DnnEventTagger(const std::string& name);
  ~DnnEventTagger();

  bool initialize();
  bool execute();
  bool finalize();

private:
  int fPitch;
};

#endif /* DnnEventTagger_hh */
