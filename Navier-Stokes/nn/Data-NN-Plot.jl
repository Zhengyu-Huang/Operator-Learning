using LinearAlgebra
using PyPlot
include("../../nn/mynn.jl")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 22
    font0 = Dict(
    "font.size" => 30,          # title
    "axes.labelsize" => 26, # axes labels
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    "lines.linewidth" => 3,
    "lines.markersize" =>10,
    )
merge!(rcParams, font0)

#  Data  | width | cost  |  Training-Time | Test-Error  |  
PCA_Data = 
[ 156    16         NaN    0.48372647854157635   1.2599131654099982;
  156    64         NaN    0.11829644507783219   0.8283551915826297;
  156    128         NaN   0.05504501200901883   0.6698802621949894;
  156    256        NaN    0.02238793236798551   0.5642631482372084;
  156    512        NaN    0.007623927126534395   0.49189887333680216;
  #  
  312    16          NaN   0.402645369580926   0.4382702606355083;
  312    64          NaN   0.11599568102017023   0.5159697592862766;
  312    128         NaN   0.04884627100697905   0.4292427006344829;
  312    256         NaN   0.018155492149574383   0.35027944942191597;
  312    512         NaN   0.006659555502235408   0.30907161017929013;
  #  
  625    16           NaN    0.3558535283096477   0.3843909207063844;
  625    64           NaN    0.10516639811931026   0.2130797895257004;  
  625    128         NaN     0.051859665709281695   0.3210348857868852;   
  625    256          NaN    0.015814735271475717   0.2693049358492665;
  625    512         NaN     0.006657290591955316   0.2302239596918842;  
  #  
  1250    16    NaN   0.3420099087349544   0.35599178436679574;
  1250    64   NaN    0.09834705928105526   0.12742421417733996;
  1250    128  NaN    0.06249765372188634   0.17498728787020276;
  1250    256  NaN    0.020200932937229975   0.20653458457787865;
  1250    512   NaN   0.007850239198299499   0.1793658295144119; 
  #  
  2500    16   NaN  0.34571702168382373   0.35100752928889317;
  2500    64   NaN  0.08697034590234573   0.10015709180569597;
  2500    128  NaN  0.050315027516566975   0.06884646208258045;
  2500    256  NaN  0.02996395165175071   0.12327805802868226;
  2500    512   NaN  0.012487493919171248   0.15053986164940739;  
  #  
  5000    16    NaN   0.3681625890001357   0.37269359623622994;
  5000    64   NaN    0.08453765405921035   0.09119488983326378;
  5000    128  NaN    0.04655351689140005   0.054835644633447214;
  5000    256  NaN    0.023096231620844954   0.040003545163146983;
  5000    512  NaN    0.0212275249357543   0.09016458251929085;  
  #  
  10000    16  NaN   0.3482440561096979   0.34813904350813396;
  10000    64  NaN   0.07520859445484773   0.07825248179447915;
  10000    128 NaN   0.04468250565988809   0.04901481059863196;
  10000    256 NaN   0.027363091016535918   0.03369944206225932;
  10000    512 NaN   0.02253369170529204   0.02738171358136215;
  #  ?????
  20000    16   NaN NaN   NaN;
  20000    64  NaN  NaN   NaN;
  20000    128 NaN NaN   NaN;
  20000    256 NaN NaN   NaN;
  20000    512 NaN NaN   NaN
]

DeepONet_Data = 
 #################################  
 [ 156    16         NaN    0.6298379515056217   1.089125657335339;
  156    64         NaN    0.22982885234630684   0.9326485151098609;
  156    128         NaN   0.13542048153611702   0.7622036621782401;
  156    256        NaN    0.08557835911367227   0.6733707666649699;
  156    512        NaN    0.1267458091689424   0.5672771801294582;
  #  
  312    16          NaN   0.6296778783501483   0.7182663158165652;
  312    64          NaN   0.20863398529102425   0.5616818233270175;
  312    128         NaN   0.10756804633930009   0.47262731279874953;
  312    256         NaN   0.06251478274333189   0.40392174505848405;
  312    512         NaN   0.08681451528783803   0.3464614604333941;
  #  
  625    16           NaN    0.5056929262724805   0.5165487368577917;
  625    64           NaN    0.18744987343260236   0.33396699005821917;  
  625    128         NaN     0.08989883103361014   0.3618419618692635;   
  625    256          NaN    0.0517193073898234   0.30261322268190616;
  625    512         NaN     0.047740440975130645   0.24579367592334234;  
  #  
  1250    16    NaN   0.48047262458313417   0.4907128952477202;
  1250    64   NaN    0.1368642320991671   0.15306434444642128;
  1250    128  NaN    0.09352853399893565   0.1998022483207115;
  1250    256  NaN    0.04795417530413867   0.21911648091013702;
  1250    512   NaN   0.05172877978204119   0.1782143748784498; 
  #  
  2500    16   NaN  0.436973841389058   0.4364298037979343;
  2500    64   NaN  0.12035760926281987   0.12533884094559097;
  2500    128  NaN  0.06743277199648617   0.07897681074307918;
  2500    256  NaN  0.04752765444480625   0.09129820963757006;
  2500    512   NaN  0.05669324280064768   0.11911321396738604;  
  #  
  5000    16    NaN   0.44690564326395177   0.44921424875254695;
  5000    64   NaN    0.10114226388680543   0.104134672319127;
  5000    128  NaN    0.058629822131333706   0.0639932401863222;
  5000    256  NaN    0.03827198024966348   0.0471576009451087;
  5000    512  NaN    0.05429819877144723   0.06487530182367197;  
  #  
  10000    16  NaN   0.3482440561096979   0.34813904350813396;
  10000    64  NaN   0.09084050779534474   0.09173889460675766;
  10000    128 NaN   0.05486353908662727   0.05762375229312928;
  10000    256 NaN   0.03655212392885742   0.040541269893922524;
  10000    512 NaN   0.05378287975827432   0.05617889504327903;
  #  ?????
  20000    16   NaN NaN   NaN;
  20000    64  NaN  NaN   NaN;
  20000    128 NaN  NaN   NaN;
  20000    256 NaN  NaN   NaN;
  20000    512 NaN  NaN   NaN
]



PARA_Data = 
 [ 156    16         NaN        0.4777425548055959   5.105722696792032
  156    64         NaN         0.16094428112676676   7.603432709641743
  156    128         NaN        0.11171983232945827   5.270794426886748
  156    256        NaN         0.052828844208461415   2.540549238464916
  156    512        NaN         0.03830589383387005   1.3013849736373164
  #  
  312    16          NaN        0.6041225142636976   1.718251623149174
  312    64          NaN        0.22185569289764187   4.986682591208396
  312    128         NaN        0.14999902550399244   4.189121192185107
  312    256         NaN        0.05735699950401425   1.8658741927176892 
  312    512         NaN        0.03795431066462735   0.9770857754604827
  #  
  625    16           NaN       0.4630247753739805   0.4901636000466877;
  625    64           NaN       0.278142295212498   1.6784329672134009;   
  625    128         NaN        0.1597787529234006   2.1557757351708946;   
  625    256          NaN       0.0564317256300798   1.250278575263224;
  625    512         NaN        0.03158571186356892   0.6840505478638857;  
  #  
  1250    16    NaN  0.46901304403691035   0.4843732312500977;
  1250    64   NaN   0.15697765494421204   0.1710303015965594;
  1250    128  NaN   0.0956253367030585   0.13373727430618426;
  1250    256  NaN   0.06982768699014055   0.6192348338857112;
  1250    512   NaN  0.0519002127621824   0.458913425407676;  
  #  
  2500    16   NaN    0.4703114600059343   0.47272605086164693;
  2500    64   NaN    0.15425868877932825   0.15865979052693605;
  2500    128  NaN    0.07996747623598974   0.08757023742621876;
  2500    256  NaN    0.06016989640306503   0.08905111859488479;
  2500    512   NaN   0.044345188933280544   0.24732868583162118;  
  #  
  5000    16    NaN  0.46749259153568745   0.47131869921409236;
  5000    64   NaN   0.14778896037466813   0.15051914694539614;
  5000    128  NaN   0.0732319472002253   0.07757344028183281;
  5000    256  NaN   0.05082868890884618   0.05708960509295496;
  5000    512  NaN   0.041940697361751086   0.0510535398264535;  
  #  
  10000    16  NaN  0.4602192834394011   0.45884623636979266;
  10000    64  NaN   0.13567151011430675   0.13623955715692415;
  10000    128 NaN   0.07219091145201005   0.07373488309277397;
  10000    256 NaN   0.043859554527029984   0.04635444361818899;
  10000    512 NaN   0.039361871151748606   0.043010155861904334
]

FNO_Data =
 [ 156    2         NaN        0.10233721074958642   0.10641503415237634;
  156    4         NaN         0.04139418834342788   0.052481138720535316;
  156    8         NaN         0.016331819375642598   0.027683810462267734;
  156    16        NaN         0.005084847297089605   0.024489252254939996;
  156    32        NaN         0.0014837636136246894   0.023107163088682752;
  #  
  312    2          NaN        0.08497613236212577   0.08785441608574146;
  312    4          NaN        0.030050702028883956   0.03543143274071507;
  312    8         NaN         0.012746667059568258   0.018383976809370022;
  312    16         NaN        0.005331554372484486   0.014313613769049063;
  312    32         NaN        0.0019025466351125103   0.011646377294360159;
  #  
  
  625    2           NaN          0.07378177320361137   0.07453825324177742;
  625    4           NaN          0.02178352489620447   0.023542389929294586;   
  625    8         NaN            0.00998089706748724   0.011948890105634928;   
  625    16          NaN          0.004523765351250768   0.0075276368826627735;
  625    32         NaN           0.001944685984030366   0.005216458012908697;  
  #  
  1250    2    NaN 0.06245762360394001   0.06286482735872269;
  1250     4   NaN 0.017422971499711275   0.018334206885099413;
  1250     8  NaN 0.007673453948646784   0.0085724423173815;
  1250    16  NaN 0.0035465318309143186   0.0045669326117262245;
  1250    32   NaN  0.0017730872983112932   0.0028392662533558905;  
  #  
  
  2500    2   NaN 0.05690229085534811   0.05757371139526367;
  2500     4   NaN 0.014440094044432044   0.014941215437278152;
  2500     8  NaN 0.005943128981534392   0.006385699169524014;
  2500    16  NaN 0.002590934175159782   0.002971284946007654;
  2500    32   NaN  0.0015480940816458314   0.0019464851193130017;  
  #  
  5000    2    NaN     0.05360847172029316   0.053726130229979754;
  5000    4   NaN      0.01269859040537849   0.012826241092197597;
  5000     8  NaN      0.004733569839317352   0.00490279544517397;
  5000    16  NaN      0.0019681449374649674   0.002116999446065165;
  5000    32  NaN      0.0011492412745603361   0.0012877331201452762;  
  #  
  10000    2  NaN      0.05087686530221253   0.051055041182786226;
  10000    4  NaN      0.010374792006704957   0.010481059829844161;
  10000    8 NaN       0.0038119951631175354   0.003890886048297398;
  10000    16 NaN      0.0015331868732988369   0.001594744636496762;
  10000    32 NaN      0.0009968394221912605   0.0010543358584924134;
  #  
  20000    2   NaN    NaN   NaN;
  20000    4   NaN    NaN   NaN;
  20000    8   NaN    NaN   NaN;
  20000    16  NaN    NaN   NaN;
  20000    32  NaN    NaN   NaN
]
######################################################







# width
nns = ["PCA-Net", "DeepO-Net", "PARA-Net", "FNO"]
sizes = [L"w = 16\,\,/\,\,d_f = 2",L"w = 64 \,\,/\,\ d_f = 4",L"w = 128 \,\,/\,\ d_f = 8",L"w = 256 \,\,/\,\ d_f = 16"]
colors = ["#3A637B", "#C4A46B", "#FF6917", "#D44141" ] # colorblind friendly pallet https://davidmathlogic.com/colorblind/#%233A637B-%23C4A46B-%23FF6917-%23D44141
markers = ["o", "s", "^", "*"]
linestyle = ["dotted", "-.", "--", "-", ]

fig, ax = PyPlot.subplots(ncols = 4, sharex=false, sharey=true, figsize=(24,6))
for i = 1:4
    ax[1].plot(PCA_Data[(i+3)*5+1:(i+3)*5+5, 2], PCA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[1], linestyle=linestyle[i], marker = markers[i], fillstyle="none",      label =  "N = "*string(Int(PCA_Data[(i+3)*5+1, 1])))
    ax[2].plot(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 2], DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[2], linestyle=linestyle[i], marker = markers[i], fillstyle="none", label =  "N = "*string(Int(DeepONet_Data[(i+3)*5+1, 1])))
    if i < 4
        ax[3].plot(PARA_Data[(i+3)*5+1:(i+3)*5+5, 2], PARA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[3], linestyle=linestyle[i], marker = markers[i], fillstyle="none",     label =  "N = "*string(Int(PARA_Data[(i+3)*5+1, 1]))  )
    end
    ax[4].plot(FNO_Data[(i+3)*5+1:(i+3)*5+5, 2], FNO_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[4], linestyle=linestyle[i], marker = markers[i], fillstyle="none",      label =  "N = "*string(Int(FNO_Data[(i+3)*5+1, 1]))  )
end


for i = 1:4
    ax[i].title.set_text(nns[i])   
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["bottom"].set_color("#808080")
    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](colors="#808080")
end
ax[1].legend(frameon=false,handlelength=3.4)
ax[1].set_ylabel("Test error")

i=1
ax[1].set_xticks(PCA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[1].set_xlabel("Network width",labelpad=20)
ax[2].set_xticks(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[2].set_xlabel("Network width",labelpad=20)
ax[3].set_xticks(PARA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[3].set_xlabel("Network width",labelpad=20)
ax[4].set_xticks(FNO_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[4].set_xlabel("Lifting dimension",labelpad=20)

plt.tight_layout()
plt.savefig("NS-Width-Error.pdf")
plt.close()



## Data vs Error
fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(24,6.5))
row_ids = [1,2,3,4]
plot2_yticks = [1e-2, 1e-1, 1, 10]
N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]
for i = 1:3 
    ax[i].loglog(N_Data, 0.1*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=1)
    ax[i].text(11000,0.015,"1/√N",color="#bababa",fontsize=22)
    lh1 = ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none")
    lh2 = ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none")
    lh3 = ax[i].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none" )
    lh4 = ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none" )
end
ax[4].loglog(N_Data, 0.1*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=1)
ax[4].text(11000,0.015,"1/√N",color="#bababa",fontsize=22)
ax[4].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",      label =  nns[1]  )
ax[4].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",      label =  nns[2]  )
ax[4].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",      label =  nns[3]  )
ax[4].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",      label =  nns[4]  )

for i = 1:4
    ax[i].set_title(sizes[i],pad=50)
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].set_xticks(N_Data[2:2:end])
    ax[i].set_xticklabels(N_Data[2:2:end])
    ax[i].set_yticks(plot2_yticks)

    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](which="minor",left=false) # remove minor ytick labels?
    ax[i].set_xlabel(latexstring("Training data ",L"N"),labelpad=20)
end
ax[1].set_yticklabels(plot2_yticks)
ax[1].set_ylabel("Test error")

fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.87),ncol=4,frameon=false)

plt.tight_layout()
plt.savefig("NS-Data-Error.pdf")
plt.close()








# complexity
Np=101*101
layers = 4
n_in = 128
kmax = 12*12
for i = 1:size(PCA_Data)[1]
    PCA_Data[i, 3] = PCA_Net_Cost(n_in, PCA_Data[i, 2],layers, Np)
end
for i = 1:size(DeepONet_Data)[1]
    DeepONet_Data[i, 3] = DeepO_Net_Cost(n_in, DeepONet_Data[i, 2],layers, Np)
end
for i = 1:size(PARA_Data)[1]
    PARA_Data[i, 3] = PARA_Net_Cost(n_in, PARA_Data[i, 2],layers, Np)
end
for i = 1:size(FNO_Data)[1]
    FNO_Data[i, 3] = FNO_Net_Cost(FNO_Data[i, 2], kmax, layers, Np)
end

fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(24,6))
for i = 1:4
    ax[i].semilogx(PCA_Data[(i+3)*5+1:(i+3)*5+5, 3], PCA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[1], linestyle=linestyle[i], marker = markers[1], fillstyle="none",      label =  nns[1]  )
    ax[i].semilogx(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 3], DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[2], linestyle=linestyle[i], marker = markers[2], fillstyle="none", label =  nns[2]  )
    
    if i < 4
        ax[i].semilogx(PARA_Data[(i+3)*5+1:(i+3)*5+5, 3], PARA_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[3], linestyle=linestyle[i], marker = markers[3], fillstyle="none",     label =  nns[3]  )
    end
    
    ax[i].semilogx(FNO_Data[(i+3)*5+1:(i+3)*5+5, 3], FNO_Data[(i+3)*5+1:(i+3)*5+5, 5], color = colors[4], linestyle=linestyle[i], marker = markers[4], fillstyle="none",      label =  nns[4]  )
    ax[i].title.set_text(L"N = "*string(Int(FNO_Data[(i+3)*5+1, 1])))   
end
for i = 1:4
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["bottom"].set_color("#808080")
    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](colors="#808080")
    ax[i].set_xticks([1e6, 1e8,1e10])
    ax[i].set_xticklabels([L"10^6",L"10^8",L"10^10"])
    ax[i].set_xlabel("Evaluation complexity",labelpad=20)
end
ax[3].legend(frameon=false,handlelength=0)
ax[1].set_ylabel("Test error")

plt.tight_layout()
plt.savefig("NS-Cost-Error.pdf")






