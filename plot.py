import pickle
import ROOT
import array

h1 = ROOT.TH1F('h1', 'h1', 100, 0, 1)
h2 = ROOT.TH1F('h2', 'h2', 100, 0, 1)
output = pickle.load( open( "save_9mm.p", "rb" ) )
score = output[4][14]
for item in score:
    score, target = float(item[0]), int(item[1])
    if target == 0:
        h1.Fill(score)
    else:
        h2.Fill(score)

c1 = ROOT.TCanvas('c1', 'c1', 800, 600)
c1.SetLogy(1)
h1.SetLineColor(ROOT.kRed)
h2.SetLineColor(ROOT.kBlue)
h1.SetMaximum(1.1*max(h1.GetMaximum(), h2.GetMaximum()))
h1.Draw()
h2.Draw('histsame')
leg1 = ROOT.TLegend(0.4, 0.6, 0.65, 0.85)
leg1.AddEntry(h1, 'Gamma','l')
leg1.AddEntry(h2, 'Electron', 'l')
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)
h1.SetXTitle('DNN output')
h1.SetYTitle('# of events')
leg1.Draw()
c1.Print('dnn_output_9mm.png')
bkg = array.array('d')
sig = array.array('d')
for i in range(h1.GetNbinsX()):
    bkg.append(h1.Integral(i,101)*1.0/h1.Integral()*0.99)
    sig.append(h2.Integral(i,101)*1.0/h2.Integral()*0.99)
def sig_eff(rootfile):
    tmva = ROOT.TFile(rootfile,'READ')
    ttree = tmva.Get('Charge/TestTree')
    eff = []
    for j in range(1, 100):
        bdtg = j*0.2
        eff.append((ttree.GetEntries('classID==1&&BDTG>%f' % bdtg)*1.0/ttree.GetEntries('classID==1'), ttree.GetEntries('classID==0&&BDTG>%f' % bdtg)*1.0/ttree.GetEntries('classID==0')))
    return eff

bdt_eff = sig_eff('tmva_pid_9mm_roi.root')
sig_bdt = array.array('d')
bkg_bdt = array.array('d')
for item in bdt_eff:
    sig_bdt.append(item[0])
    bkg_bdt.append(item[1])

for bkgi, sigi in zip(bkg, sig):
    print bkgi, sigi
roc = ROOT.TGraph(len(bkg), bkg, sig)
roc_bdt = ROOT.TGraph(len(bkg_bdt), bkg_bdt, sig_bdt)
c2 = ROOT.TCanvas('c2', 'c2', 800,600)
cf = c2.DrawFrame(0, 0.4, 0.1, 1)
cf.SetXTitle('bkg misID')
cf.SetYTitle('signal efficiency')
roc.Draw('cp')
roc.SetLineColor(ROOT.kRed)
roc_bdt.Draw('cp')
roc_bdt.SetLineColor(ROOT.kBlue)
leg2 = ROOT.TLegend(0.5, 0.2, 0.75, 0.4)
leg2.AddEntry(roc, 'DNN', 'l')
leg2.AddEntry(roc_bdt, 'BDT', 'l')
leg2.Draw()
c2.Print('dnn_roc_9mm_bdt.png')
