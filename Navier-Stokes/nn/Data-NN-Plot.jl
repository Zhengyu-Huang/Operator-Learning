using LinearAlgebra
using PyPlot
include("../../nn/mynn.jl")
include("../../plotdefaults.jl")


#  Data  | width | cost  |  Training-Time | Test-Error  |  
PCA_Data = 
[ 156    16         NaN    0.4605269440790674   1.2186841774199568;
  156    64         NaN    0.12295705740592087   0.8310629670468895;
  156    128         NaN   0.053029945474354405   0.6645750325474368;
  156    256        NaN    0.024155865746088735   0.5354438089860011;
  156    512        NaN    0.007685656782677317   0.49707179832029413;
  #  
  312    16          NaN   0.4061468138812397   0.46896098348017734;
  312    64          NaN   0.11849051304339152   0.5336079010235679;
  312    128         NaN   0.04880362119750032   0.44335853513590945;
  312    256         NaN   0.0175888082263867   0.35221964021087676;
  312    512         NaN   0.0065256781489666914   0.3118061568071539;
  #  
  625    16           NaN    0.363093692415487   0.3773597898456187;
  625    64           NaN    0.10687483216882657   0.2226816340754798;  
  625    128         NaN     0.05057964963015008   0.3275554870351507;   
  625    256          NaN    0.01581713332476597   0.2719887937186588;
  625    512         NaN     0.006539251746777073   0.22988584363873793;  
  #  
  1250    16    NaN   0.34239901203976525   0.35394876213335447;
  1250    64   NaN    0.09659145914305145   0.12568044962620772;
  1250    128  NaN    0.05891511877191601   0.16230037447526588;
  1250    256  NaN    0.020469093310919236   0.2059543400959292;
  1250    512   NaN   0.007707235087991482   0.18018489432201865; 
  #  
  2500    16   NaN  0.3439587012689532   0.349524148546195;
  2500    64   NaN  0.08516474312213532   0.09942572347111878;
  2500    128  NaN  0.05081069150591965   0.06952146581640119;
  2500    256  NaN  0.029418206369017742   0.12722447300083467;
  2500    512   NaN  0.012571926083063841   0.15107817267176432;  
  #  todo
  5000    16    NaN   0.344605033726457   0.3514301322913124;
  5000    64   NaN    0.0806649869215545   0.08787424285017831;
  5000    128  NaN    0.04563902266467284   0.05441332747279016;
  5000    256  NaN    0.022890382255635264   0.0399625735269843;
  5000    512  NaN    0.020807149058156456   0.09732157680134786;  
  #  
  10000    16  NaN   0.34710749574442873   0.3511476522044054;
  10000    64  NaN   0.07669613822195874   0.07983375349818746;
  10000    128 NaN   0.045445755624857066   0.050102329075302314;
  10000    256 NaN   0.02763190948655302   0.034348684443923375;
  10000    512 NaN   0.024056045138034488   0.028777487401718076;
  #  ?????
  20000    16 NaN  0.34813341022140537   0.34936359398047145;
  20000    64 NaN  0.0720341307586599   0.07377127531598891;
  20000    128 NaN 0.041655531357016676   0.04403831596905797;
  20000    256 NaN 0.030945830370406907   0.033546081579739;
  20000    512 NaN 0.024654289891594947   0.02654266348914689
]

DeepONet_Data = 
 #################################  
 [ 156    16         NaN    0.6823236037995135   1.0965994258088223;
  156    64         NaN     0.22332388843052683   0.8685441901822833;
  156    128         NaN    0.13788389818369423   0.7217327642810997;
  156    256        NaN     0.0926387926888328   0.6270829225776512;
  156    512        NaN     0.11959660024045686   0.6124303985625325;
  #  
  312    16          NaN    0.5920918415620254   0.7132703193464228;
  312    64          NaN    0.20438609611081196   0.5916774049906257;
  312    128         NaN    0.10547281394364656   0.4873756682449709;
  312    256         NaN    0.06358655652570984   0.39945583181483313;
  312    512         NaN    0.08633713718451563   0.3363341079048183;
  #  
  625    16           NaN    0.505488787180168   0.5125248732928046;
  625    64           NaN    0.18731977969516037   0.321295130258568;  
  625    128         NaN     0.08938945211160969   0.36520905045410174;   
  625    256          NaN    0.049705610314521066   0.29954551132510143;
  625    512         NaN     0.04729276596314312   0.24727751635001988;  
  #  
  1250    16    NaN   0.48375488633763053   0.48802059716270585;
  1250    64   NaN    0.1463045225320293   0.16121094726135068;
  1250    128  NaN    0.09054544032860942   0.18562608901755462;
  1250    256  NaN    0.05281790836345452   0.21806865859666133;
  1250    512   NaN   0.04544998746006488   0.17450553353410342; 
  #  
  2500    16   NaN  0.4449047468628387   0.4483444513676299;
  2500    64   NaN  0.11664017523614394   0.12329129833723948;
  2500    128  NaN  0.0651567139043255   0.0783663627736766;
  2500    256  NaN  0.04941679620912793   0.10260458796460414;
  2500    512   NaN  0.046842023801196356   0.11638213968757355;  
  #  
  5000    16    NaN   0.44060786051061207   0.4468531587113586;
  5000    64   NaN    0.10384854525693495   0.10765609765628274;
  5000    128  NaN    0.05923312147415374   0.06496502959565158;
  5000    256  NaN    0.03945328447453681   0.04864137469300518;
  5000    512  NaN    0.0679997757933887   0.07808885045365166;  
  #  
  10000    16  NaN   0.4408954004181099   0.4437397361818173;
  10000    64  NaN   0.09654046071748165   0.0982182973525812;
  10000    128 NaN   0.05224020509405385   0.05517650146380279;
  10000    256 NaN   0.03677390596071996   0.04110120931348545;
  10000    512 NaN   0.05021867729921666   0.05299066989203525;
  #  ?????
  20000    16   NaN  0.41246251132533335   0.4121163467766536;
  20000    64  NaN   0.0870199999205234   0.08764693351758904;
  20000    128 NaN   0.05090797087191457   0.05247415834110915;
  20000    256 NaN   0.034273759354919914   0.03629471638135709;
  20000    512 NaN   0.04030919341495455   0.04193620966281372
]



PARA_Data = 
 [ 156    16         NaN        0.4817009113503805   4.26485847731814;
  156    64         NaN         0.15920813520008936   8.016100511498273;
  156    128         NaN        0.09173316487379266   5.252461691140093;
  156    256        NaN         0.06186147816100523   2.475510719407746;
  156    512        NaN         0.07237383394187444   1.3392683749715506;
  #  
  312    16          NaN        0.579911428751994   1.7362415424110673;
  312    64          NaN        0.219174605677275   5.573088372403878;
  312    128         NaN        0.17098124221618433   3.9123777827317143;
  312    256         NaN        0.07398615947722617   1.8569723013434924;
  312    512         NaN        0.04189733170414732   0.9556458355900904;
  #  
  625    16           NaN       0.4718089797165145   0.49220149695729426;
  625    64           NaN       0.2903884192431171   2.039897951762062;   
  625    128         NaN        0.15344959861995244   2.258432924131567;   
  625    256          NaN       0.06890992408766512   1.2584854263994432;
  625    512         NaN        0.05196309891415873   0.7242116360031626;  
  #  
  1250    16    NaN  0.4746595258273534   0.485363079664304;
  1250    64   NaN   0.16129120267962568   0.1732828559992677;
  1250    128  NaN   0.13037294748959447   0.16637627915802317;
  1250    256  NaN   0.06533727337450614   0.6081945478692418;
  1250    512   NaN  0.03588343142822981   0.4439861268633558;  
  #  
  2500    16   NaN    0.4670612495076526   0.47109843025515186;
  2500    64   NaN    0.1520547705357872   0.15752028896254378;
  2500    128  NaN    0.08855955326435906   0.09651611101764256;
  2500    256  NaN    0.06290938134926248   0.09567441332150982;
  2500    512   NaN   0.060472142605845   0.2532183624901977;  
  #  
  5000    16    NaN  0.4655672815069447   0.47135633674639776;
  5000    64   NaN   0.15383802424202714   0.15735269684234243;
  5000    128  NaN   0.07426424418497594   0.07884265809298366;
  5000    256  NaN   0.04674108623915223   0.05425594791646079;
  5000    512  NaN   0.04379936653832022   0.05300132757279298;  
  #  
  10000    16  NaN   0.45759987791767087   0.46019298000329584;
  10000    64  NaN   0.13678016887620187   0.13804982738975685;
  10000    128 NaN   0.07564328630871214   0.07750627701519851;
  10000    256 NaN   0.04446516979231643   0.047193449539462674;
  10000    512 NaN   0.03741249818351838   0.0408874673219641
]

FNO_Data =
 [ 156    2         NaN        0.11521164270547721   0.12149539389289342;
  156    4         NaN         0.04327309332214869   0.055475699595915966;
  156    8         NaN         0.016168724602231614   0.027459482160898354;
  156    16        NaN         0.005249500053767593   0.024354932590937003;
  156    32        NaN         0.0014658996675653048   0.023554511481705002;
  #  
  312    2          NaN        0.08447800151621684   0.08886375304502554;
  312    4          NaN        0.029398753027168986   0.0348455606159778;
  312    8         NaN         0.012217864979249544   0.017588141633985706;
  312    16         NaN        0.005431252079478537   0.014073804226847222;
  312    32         NaN        0.0017822225416192594   0.012318734760181261;
  #  
  625    2           NaN       0.07806984456181526   0.07862457550764083;
  625    4           NaN       0.02227312042117119   0.023793637055158617;   
  625    8         NaN         0.01030871114730835   0.01247987933307886;   
  625    16          NaN       0.00446023159623146   0.007146189975738525;
  625    32         NaN        0.0022813551280647516   0.00559703535027802;  
  #  
  1250    2    NaN 0.06780118720531464   0.06844939381182194;
  1250     4   NaN 0.017018680108338593   0.017881662148982288;
  1250     8  NaN  0.0079419405143708   0.008845399209856986;
  1250    16  NaN  0.0035085849748924375   0.004524045167118311;
  1250    32   NaN  0.0017741449317894877   0.002914417572133243;  
  #  
  2500    2   NaN  0.057117848440259696   0.05740854775607586;
  2500     4   NaN 0.014145546003431082   0.014575068874657154;
  2500     8  NaN  0.005937939846143127   0.0063536608582362535;
  2500    16  NaN  0.002793993064202368   0.0031536878041457383;
  2500    32   NaN 0.0014087061800761148   0.0018313485201448202;  
  #  
  5000    2    NaN     0.053494763150066134   0.05357691730558872;
  5000    4   NaN      0.012669489195756615   0.012801035669073462;
  5000     8  NaN      0.00488006997150369   0.005036927171051502;
  5000    16  NaN      0.002081191560672596   0.002217856671055779;
  5000    32  NaN      0.001144796449947171   0.0012749874025234022;  
  #  
  10000    2  NaN      0.046579971103183924   0.046694104697555305;
  10000    4  NaN      0.01040249362718314   0.010492198252817615;
  10000    8 NaN       0.003864795654895715   0.003933981263567693;
  10000    16 NaN      0.0015681422499590554   0.0016250672667869367;
  10000    32 NaN      0.0008914144008886069   0.0009516634211351629;
  #  
  20000    2   NaN    0.04892155617615208   0.049128379094041885;
  20000    4   NaN    0.008901817452069373   0.008960032273083925;
  20000    8   NaN    0.003021176399633987   0.0030612854907056315;
  20000    16  NaN    0.001268567710276693   0.0012972540668648434;
  20000    32  NaN    0.0008335973911860492   0.0008595376518933335
]
######################################################







# width

fig, ax = PyPlot.subplots(ncols = 4, sharex=false, sharey=true, figsize=nn_linefigsize)
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
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
end
ax[1].legend(frameon=false,handlelength=3.4)
ax[1].set_ylabel("Test error")

i=1
ax[1].set_xticks(PCA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[1].set_xlabel("Network width "*L"w",labelpad=2)
ax[2].set_xticks(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[2].set_xlabel("Network width "*L"w",labelpad=2)
ax[3].set_xticks(PARA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[3].set_xlabel("Network width "*L"w",labelpad=2)
ax[4].set_xticks(FNO_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[4].set_xlabel("Lifting dimension "*L"d_f",labelpad=2)

plt.tight_layout()
plt.savefig("NS-Width-Error.pdf")
plt.close()



## Data vs Error
fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=nn_linefigsize)
row_ids = [1,2,3,4]
plot2_yticks = [1e-2, 1e-1, 1, 10]
N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]
for i = 1:3 
    ax[i].loglog(N_Data, 0.1*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=0.5)
    ax[i].text(11000,0.015,"1/√N",color="#bababa",fontsize=6)
    lh1 = ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none")
    lh2 = ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none")
    lh3 = ax[i].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none" )
    lh4 = ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none" )
end
i = 4
ax[4].loglog(N_Data, 0.08*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=0.5)
ax[4].text(11000,0.012,"1/√N",color="#bababa",fontsize=6)
ax[4].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",      label =  nns[1]  )
ax[4].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",      label =  nns[2]  )
ax[4].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",      label =  nns[3]  )
ax[4].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",      label =  nns[4]  )

for i = 1:4
    ax[i].set_title(sizes[i],pad=10)
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i].set_xticks(N_Data[2:2:end])
    ax[i].set_xticklabels(N_Data[2:2:end])
    ax[i].set_yticks(plot2_yticks)

    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](which="minor",left=false) # remove minor ytick labels?
    ax[i].set_xlabel(latexstring("Training data ",L"N"),labelpad=2)
end
ax[1].set_yticklabels(plot2_yticks)
ax[1].set_ylabel("Test error")

fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.87),ncol=4,frameon=false)

plt.tight_layout()
plt.savefig("NS-Data-Error.pdf")
plt.close()





# complexity
Np=64*64
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

fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=nn_linefigsize)
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
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i].set_xticks([1e6, 1e8,1e10])
    ax[i].set_xticklabels([L"10^6",L"10^8",L"10^{10}"])
    ax[i].set_xlabel("Evaluation complexity",labelpad=2)
end
ax[3].legend(frameon=false,handlelength=0)
ax[1].set_ylabel("Test error")

plt.tight_layout()
plt.savefig("NS-Cost-Error.pdf")






