using LinearAlgebra
using PyPlot
include("../../nn/mynn.jl")
include("../../plotdefaults.jl")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 7
    font0 = Dict(
    "font.size" => 10,          # title
    "axes.labelsize" => 8, # axes labels
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    "lines.linewidth" => 0.7,
    "lines.markersize" =>2.5,
    )
merge!(rcParams, font0)

#  Data  | width | cost  |  Training-Time | Test-Error  |  
PCA_Data = 
[ 156    16         NaN    0.41148450443381096   1.625810339255293;
  156    64         NaN    0.16496452282641438   1.0629169437923072;
  156    128        NaN    0.09474028821582185   0.9356175810089464;
  156    256        NaN    0.031740080927560484   0.8089603786889419;
  156    512        NaN    0.0032491146734920985   0.739368829983151;
  #  
  312    16          NaN   0.35250528945185067   1.1101436777544917;
  312    64          NaN   0.16151865699318377   0.7631874803789671;
  312    128         NaN   0.09640880463645427   0.6478265381605203;
  312    256         NaN   0.039790927726516344   0.5858922346261116;
  312    512         NaN   0.008742270185856497   0.5330069291528844;
  #  
  625    16           NaN   0.31609561130602726   0.4520118750684096;
  625    64           NaN   0.16324623812752087   0.4057864120311623;  
  625    128          NaN   0.10051790706655708   0.3723980936905896;   
  625    256          NaN   0.04791087091548228   0.3433764106426739;
  625    512          NaN   0.011116839599901868   0.30930197886407357;  
  #  
  1250    16   NaN    0.2602626102718969   0.2658266222693516;
  1250    64   NaN    0.1556995448013612   0.19852913813760492;
  1250    128  NaN    0.11249045256776424   0.22878979751720818;
  1250    256  NaN    0.06274138235999062   0.23682843428824227;
  1250    512  NaN    0.02310467444972777   0.21220755350122175; 
  #  
  2500    16   NaN  0.2911999786587446   0.2955836267903681;
  2500    64   NaN  0.1447699662173425   0.15931452747039812;
  2500    128  NaN  0.11231850175061957   0.16142522843480409;
  2500    256  NaN  0.07510072071136398   0.18567479104726775;
  2500    512  NaN  0.02695173948095355   0.17563196034495848;  
  #  
  5000    16   NaN   0.2704528647550728   0.27169091853385946;
  5000    64   NaN   0.1506546960289392   0.15765421858879866;
  5000    128  NaN   0.11084497756644236   0.13906545950753305;
  5000    256  NaN   0.07371837976065146   0.16756864533451618;
  5000    512  NaN   0.021291121921620602   0.1527408494549857;  
  #  
  10000    16  NaN   0.2707286331343262   0.27075248693298576;
  10000    64  NaN   0.14286711040001046   0.14600134229109132;
  10000    128 NaN   0.11268536549659852   0.1313018292445184;
  10000    256 NaN   0.06250115808618784   0.16160805957839333;
  10000    512 NaN   0.030635775419660706   0.15176696266648593;
  #  
  20000    16  NaN   0.2609869382039488   0.26137727003888633;
  20000    64  NaN   0.14509676681568734   0.14685110188186196;
  20000    128 NaN   0.11496675471775764   0.12531453570808304;
  20000    256 NaN   0.06300039146761455   0.1615296879140314;
  20000    512 NaN   0.044766228513230574   0.153849630330182
]

DeepONet_Data = 
 #################################  
 [ 156    16         NaN     0.5070580248489251   1.6189549263542216;
  156    64         NaN      0.32871913156023735   1.05655180168334;
  156    128         NaN     0.2606783749183887   0.9833009532472692;
  156    256        NaN      0.22213076778154234   0.7997425615738476;
  156    512        NaN      0.21050424108552226   0.812098064989847;
  #  
  312    16          NaN     0.4828297177863973   0.8937357900524027;
  312    64          NaN     0.2970174212435104   0.7599257161720466;
  312    128         NaN     0.23086576840032252   0.6293243710790005;
  312    256         NaN     0.18494241982334578   0.589066222274057;
  312    512         NaN     0.19701723237384777   0.5579974212453305;
  #  
  625    16           NaN    0.43298696261339126   0.5313266016900908;
  625    64           NaN    0.2631497661524192   0.4134477063990766;   
  625    128          NaN    0.20069787008685003   0.3675204990011779;   
  625    256          NaN    0.17377767621363227   0.341175346198033;
  625    512          NaN    0.16269238388969665   0.3079901534920736;  
  #  
  1250    16   NaN    0.42307395906430395   0.4280699948288257;
  1250    64   NaN    0.23294048914977353   0.2677328089642323;
  1250    128  NaN    0.17600211564692708   0.23835284569033693;
  1250    256  NaN    0.15419882031211463   0.22262201754015723;
  1250    512   NaN   0.14989769193582542   0.20782632134121376;  
  #  
  2500    16   NaN        0.34443226320742926   0.34953051365985766;
  2500    64   NaN        0.20569867654672117   0.2184996732720151;
  2500    128  NaN        0.1635424020233522   0.1923301554094833;
  2500    256  NaN        0.13727318342908684   0.1826352111262968;
  2500    512   NaN       0.142441122510508   0.17971471014935514;  
  #  
  5000    16    NaN  0.3390015311176906   0.339854837990065;
  5000    64   NaN   0.19125903806697445   0.19462081525486363;
  5000    128  NaN   0.15178129141255828   0.17090106431329896;
  5000    256  NaN   0.1256369182009571   0.1629317373210825;
  5000    512  NaN   0.12829018558872063   0.16149004169295325;  
  #  
  10000    16  NaN  0.31485445315516086   0.3147192442972554;
  10000    64  NaN  0.1754354707417321   0.17691830407099424;
  10000    128 NaN  0.13959321318814621   0.1595090071011537;
  10000    256 NaN  0.12371872580493146   0.15864968584320482;
  10000    512 NaN  0.1186759055943948   0.1524074389194938;
  #  
  20000    16   NaN  0.2818526103174339   0.2824034516936106;
  20000    64  NaN   0.17532498957747206   0.17641142396665413;
  20000    128 NaN   0.1412348384360671   0.1534566701805047;
  20000    256 NaN   0.12131616487757463   0.15380293653956004;
  20000    512 NaN   0.126880080082424   0.1540948772027392
]



PARA_Data = 
 [ 156    16         NaN        0.26762557309627255   10.567036352369554;
  156    64         NaN         0.22871510111268242   5.051620374844744;
  156    128         NaN        0.22316206109768727   2.428105823731627;
  156    256        NaN         0.2037239725826691   1.247770079930493;
  156    512        NaN         0.19134236082736825   1.2310723634497105;
  #  
  312    16          NaN        0.34807542249653095   2.56904571069103;
  312    64          NaN        0.22759407907858623   1.8656539839038173;
  312    128         NaN        0.2228739292529412   1.2377686059868398;
  312    256         NaN        0.19699755814187636   0.9336052197877641;
  312    512         NaN        0.18817812997884098   0.8444843272033618;
  #  
  625    16           NaN       0.22156512484489194   0.849356584652454;
  625    64           NaN       0.19877872260630947   0.8350517037576246;   
  625    128         NaN        0.2033529973759153   0.7162350142261116;   
  625    256          NaN       0.17583636190051297   0.5982148373186592;
  625    512         NaN        0.17841499435441988   0.5895871621781661;  
  #  
  1250    16    NaN      0.21027730502993128   0.4907082794919668;
  1250    64   NaN       0.16518972108805097   0.52652695437646;
  1250    128  NaN       0.16261006510116932   0.4575275881074447;
  1250    256  NaN       0.1647856129194096   0.4064735011136244;
  1250    512   NaN      0.16059090954537034   0.40761957854180036;  
  #  
  2500    16   NaN    0.20639510082715798   0.32398266195456715;
  2500    64   NaN    0.14981098224826628   0.3815895700807757;
  2500    128  NaN    0.13859260792424988   0.3253177858931568;
  2500    256  NaN    0.1376911985317166   0.31370981848926477;
  2500    512   NaN   0.1400510848122096   0.30180196308948837;  
  #  
  5000    16    NaN  0.19055711923265867   0.22876288245332427;
  5000    64   NaN   0.140744880128135   0.26782993190745713;
  5000    128  NaN   0.13022641055224213   0.25138773938080056;
  5000    256  NaN   0.12813148382959771   0.249383753223002;
  5000    512  NaN   0.11914794570625588   0.233556070282211;  
  #  
  10000    16  NaN    0.17818617510566212   0.18938821704508652;
  10000    64  NaN   0.13537191187042774   0.20500198815404946;
  10000    128 NaN   0.1179429144528301   0.2133090602045811;
  10000    256 NaN   0.11286498847465651   0.20326229111377805;
  10000    512 NaN   0.11133452466670658   0.19839667703581082;
  #
  20000    16  NaN   0.1657959197563843   0.1713110576132001;
  20000    64  NaN   0.1341554075674926   0.16459538628063677;
  20000    128 NaN   0.11983618958668557   0.17752374839594073;
  20000    256 NaN   0.10621718390607414   0.17848990209261975;
  20000    512 NaN   0.1049167472534716   0.17596032934323255;
]

FNO_Data =
 [ 156    2         NaN        0.14732429475929493   0.15614366803604823;
  156    4         NaN         0.13883701406228235   0.15625581274238917;
  156    8         NaN         0.1273765354297864   0.1544287418230222;
  156    16        NaN         0.11244011112751487   0.16062647889917478;
  156    32        NaN         0.08686964705180472   0.1717923815624836;
  #  
  312    2          NaN        0.20176373412593818   0.20020753412674636;
  312    4          NaN        0.13686082367665875   0.14025802174821878;
  312    8         NaN         0.1292798470419187   0.14126543063097274;
  312    16         NaN        0.10984872052302727   0.15076490514314708;
  312    32         NaN        0.09150851065388475   0.15488419201201162;
  
  #  
  625    2    NaN      0.13924972246289252   0.14289348157644272;
  625     4   NaN      0.13491478478312494   0.14151581405997277;
  625     8  NaN       0.12824824439585208   0.14251263695955277;
  625    16  NaN       0.1131444073677063   0.1498719649106264;
  625    32   NaN      0.09237072032764554   0.15386472637057305;  
  #  
  1250    2    NaN      0.16392094716429711   0.16110907166600227;
  1250     4   NaN      0.13518089597225189   0.13673287317752839;
  1250     8  NaN       0.1289861882865429   0.13826243783384562;
  1250    16  NaN       0.11087910863608122   0.14987313005924224;
  1250    32   NaN      0.09369939167052507   0.15524319107010962;  
  #  
  2500    2   NaN        0.13731948798447846   0.13936137055307626;
  2500     4   NaN       0.13437189842760563   0.13882112728506327;
  2500     8  NaN        0.1303224521882832   0.1394463504396379;
  2500    16  NaN        0.11557252415716648   0.1517387391217053;
  2500    32   NaN       0.09624609385039658   0.1625155703091994;  
  #  
  5000    2    NaN        0.14038108449429273   0.14101089098751546;
  5000    4   NaN          0.13588077685385944   0.13641340799331664;
  5000     8  NaN         0.13240791840329766   0.13632852494865655;
  5000    16  NaN         0.1215412126550451   0.14519446346964687;
  5000    32  NaN         0.10048962573166936   0.1583777679445222;  
  #  
  10000    2  NaN        0.15173418428078295   0.15114286270216107;
  10000    4  NaN        0.13454464524164797   0.134831189160794;
  10000    8 NaN         0.13284966098852455   0.13485323708280922;
  10000    16 NaN        0.1232320523605682   0.14305093303266914;
  10000    32 NaN        0.10523619120083749   0.15444058109507897;
  #  
  20000    2   NaN       0.13669491227064282   0.13680581047814339;
  20000    4   NaN       0.13373905710428954   0.13398841593470423;
  20000    8   NaN       0.13262592964693903   0.13358469334952533;
  20000    16  NaN       0.12837139420127497   0.1373032422539778;
  20000    32  NaN       0.11978913526094984   0.14484920839183033
]
######################################################



# width

fig, ax = PyPlot.subplots(ncols = 4, sharex=false, sharey=true, figsize=nn_linefigsize)
for i = 1:4
    ax[1].plot(PCA_Data[(i+3)*5+1:(i+3)*5+5, 2], PCA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[1], linestyle=linestyle[i], marker = markers[i], fillstyle="none",      label =  L"N = "*string(Int(PCA_Data[(i+3)*5+1, 1])))
    ax[2].plot(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 2], DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[2], linestyle=linestyle[i], marker = markers[i], fillstyle="none", label =  L"N = "*string(Int(DeepONet_Data[(i+3)*5+1, 1])))
    ax[3].plot(PARA_Data[(i+3)*5+1:(i+3)*5+5, 2], PARA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[3], linestyle=linestyle[i], marker = markers[i], fillstyle="none",     label =  L"N = "*string(Int(PARA_Data[(i+3)*5+1, 1]))  )
    ax[4].plot(FNO_Data[(i+3)*5+1:(i+3)*5+5, 2], FNO_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[4], linestyle=linestyle[i], marker = markers[i], fillstyle="none",      label =  L"N = "*string(Int(FNO_Data[(i+3)*5+1, 1]))  )
end
for i = 1:4
    ax[i].title.set_text(nns[i])   
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
end
ax[4].legend(frameon=false,handlelength=3.4,fontsize=7)
ax[1].set_ylabel("Test error")

i=1
ax[1].set_xticks(PCA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[1].set_xticklabels(["16","","128","256","512"])
ax[1].set_xlabel("Network width "*L"w",labelpad=2)
ax[2].set_xticks(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[2].set_xticklabels(["16","","128","256","512"])
ax[2].set_xlabel("Network width "*L"w",labelpad=2)
ax[3].set_xticks(PARA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[3].set_xticklabels(["16","","128","256","512"])
ax[3].set_xlabel("Network width "*L"w",labelpad=2)
ax[4].set_xticks(FNO_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[4].set_xlabel("Lifting dimension "*L"d_f",labelpad=2)

plt.subplots_adjust(bottom=0.22,top=0.85,left=0.07,right=0.98)
plt.savefig("Advection-dc-low-Width-Error.pdf")




## Data vs Error
fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=nn_linefigsize)
row_ids = [1,2,3,4]
# small

N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]
for i = 1:3
    ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none")
    ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none")
    ax[i].loglog(N_Data, PARA_Data[row_ids[i]:5:40, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none")
    ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none")
    
    ax[i].loglog(N_Data, 10*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=0.5)
    ax[i].text(11000,1.5,"1/√N",color="#bababa",fontsize=7)
end
ax[4].loglog(N_Data, 10*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=0.5)
ax[4].text(11000,1.5,"1/√N",color="#bababa",fontsize=7)
ax[4].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",      label =  nns[1]  )
ax[4].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",      label =  nns[2]  )
ax[4].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",      label =  nns[3]  )
ax[4].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",      label =  nns[4]  )

for i = 1:4
    ax[i].set_title(sizes[i],pad=11,fontsize=11)
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i].set_xticks(N_Data[2:2:end])
    ax[i].set_xticklabels(N_Data[2:2:end])
    # ax[i].set_yticks(plot2_yticks)

    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](which="minor",left=false) # remove minor ytick labels?
    ax[i].set_xlabel(latexstring("Training data ",L"N"),labelpad=2)
end
# ax[1].set_yticklabels(plot2_yticks)
ax[1].set_ylabel("Test error")

fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.92),ncol=4,frameon=false,fontsize=8)
plt.subplots_adjust(bottom=0.2,top=0.8,left=0.07,right=0.98)
plt.savefig("Advection-dc-low-Data-Error.pdf")





# complexity
Np=200
layers = 4
n_in = 200
kmax = 12
for i = 1:size(PCA_Data)[1]
    PCA_Data[i, 3] = PCA_Net_Cost(n_in, PCA_Data[i, 2],layers, Np)
end
for i = 1:size(DeepONet_Data)[1]
    DeepONet_Data[i, 3] = DeepO_Net_Cost(n_in, DeepONet_Data[i, 2],layers, Np)
end
for i = 1:size(PARA_Data)[1]
    PARA_Data[i, 3] = PARA_Net_Cost(n_in, 1, PARA_Data[i, 2],layers, Np)
end
for i = 1:size(FNO_Data)[1]
    FNO_Data[i, 3] = FNO_Net_Cost(FNO_Data[i, 2], kmax, layers, Np)
end

fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=nn_linefigsize)


for i = 1:4
    ax[i].semilogx(PCA_Data[(i+3)*5+1:(i+3)*5+5, 3], PCA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[1], linestyle=linestyle[i], marker = markers[1], fillstyle="none",      label =  nns[1]  )
    ax[i].semilogx(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 3], DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[2], linestyle=linestyle[i], marker = markers[2], fillstyle="none", label =  nns[2]  )
    ax[i].semilogx(PARA_Data[(i+3)*5+1:(i+3)*5+5, 3], PARA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[3], linestyle=linestyle[i], marker = markers[3], fillstyle="none",     label =  nns[3]  )
    ax[i].semilogx(FNO_Data[(i+3)*5+1:(i+3)*5+5, 3], FNO_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[4], linestyle=linestyle[i], marker = markers[4], fillstyle="none",      label =  nns[4]  )
 ax[i].title.set_text("N = "*string(Int(FNO_Data[(i+3)*5+1, 1])))   
end

for i = 1:4
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    # ax[i].set_xticks([1e5, 1e7,1e9])
    # ax[i].set_xticklabels([L"10^5",L"10^7",L"10^9"])
    ax[i].set_xlabel("Evaluation cost",labelpad=2)
end
ax[4].legend(frameon=false,handlelength=0)
ax[1].set_ylabel("Test error")


plt.subplots_adjust(bottom=0.22,top=0.85,left=0.07,right=0.98)
plt.savefig("Advection-dc-low-Cost-Error.pdf")






