using LinearAlgebra
using PyPlot
include("../../nn/mynn.jl")
include("../../plotdefaults.jl")

# rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
#     mysize = 20
#     font0 = Dict(
#     "font.size" => mysize,
#     "axes.labelsize" => mysize,
#     "xtick.labelsize" => mysize,
#     "ytick.labelsize" => mysize,
#     "legend.fontsize" => mysize,
#     )
# merge!(rcParams, font0)


#  Data  | width | cost  |  Training-Time | Test-Error  |  
PCA_Data = 
[ 156    16         NaN    0.5972945454288486   1.6023016899435356;
  156    64         NaN    0.2712941872828438   1.3267131615881533;
  156    128        NaN    0.12002444334623327   1.070313931370163;
  156    256        NaN    0.02186829823712032   0.8899788781003938;
  156    512        NaN    0.0044999892262711505   0.7780465535590242;
  #  
  312    16          NaN   0.5458528233117264   1.3126633928091698;
  312    64          NaN   0.2950012037706936   1.0251584095507376;
  312    128         NaN   0.14811131256839077   0.9091117277412162;
  312    256         NaN   0.03217800073196407   0.7626185742749478;
  312    512         NaN   0.005540517937067067   0.6483864693088186;
  #  
  625    16           NaN   0.5030848544169099   0.5307576690190434;
  625    64           NaN   0.31565612796633596   0.5654323469299223;  
  625    128          NaN   0.19032302952739888   0.5993660952925391;   
  625    256          NaN   0.05125504877129447   0.5843922644367603;
  625    512          NaN   0.0070822815655285296   0.4848240360934418;  
  #  
  1250    16   NaN    0.4819304839961137   0.4955224815272436;
  1250    64   NaN    0.3049790479894332   0.3585757263856815;
  1250    128  NaN    0.2242218822470225   0.4041493373894958;
  1250    256  NaN    0.1160853685950664   0.5120475941270206;
  1250    512  NaN    0.009894253009715191   0.45459723677737784; 
  #  
  2500    16   NaN  0.49550133610824176   0.5004977297872459;
  2500    64   NaN  0.30945250326427043   0.3315538538307341;
  2500    128  NaN  0.23574572185606393   0.2985182483290988;
  2500    256  NaN  0.1677474758009857   0.40676242068882557;
  2500    512  NaN  0.037718662484952495   0.4889634790642253;  
  #  
  5000    16   NaN   0.4964264128858891   0.49699604717150897;
  5000    64   NaN   0.3119036485703266   0.3202282981895673;
  5000    128  NaN   0.24176564170184925   0.26936589219445906;
  5000    256  NaN   0.20095225693045254   0.3130088429190938;
  5000    512  NaN   0.10567126045849841   0.4371028014419724;  
  #  
  10000    16  NaN   0.48551130819541777   0.4889537198380346;
  10000    64  NaN   0.30421665714393376   0.31199001776944013;
  10000    128 NaN   0.2450548175320096   0.26139362076843164;
  10000    256 NaN   0.21215204724871892   0.28328607013305557;
  10000    512 NaN   0.14608780466888835   0.37900160332448835;
  #  
  20000    16  NaN   0.4868503141688675   0.48801232540022726;
  20000    64  NaN   0.32087181289137806   0.3235492043899277;
  20000    128 NaN   0.2519587094353663   0.25936635191806706;
  20000    256 NaN   0.2296767577465606   0.25807651330702397;
  20000    512 NaN   0.1981099565255799   0.298869159630823
]

DeepONet_Data = 
 #################################  
 [ 156    16         NaN     0.7049315796412942   1.429108930086764;
  156    64         NaN      0.5480207160564019   1.0736163975589648;
  156    128         NaN     0.4689271285919301   0.9522123028016554;
  156    256        NaN      0.38646760311968426   0.8700827135560436;
  156    512        NaN      0.3897577934117962   0.8361723433427594;
  #  
  312    16          NaN     0.653605250580546   1.0969859542236515;
  312    64          NaN     0.48919051911886796   0.8180300463450987;
  312    128         NaN     0.4275051015189193   0.7503035019743438;
  312    256         NaN     0.36386817315856074   0.722370990247012;
  312    512         NaN     0.3572486996162347   0.6626931730601537;
  #  
  625    16           NaN    0.6431571640428725   0.6961475933839527;
  625    64           NaN    0.4760250009440566   0.5989300014593378;   
  625    128          NaN    0.3919682214412335   0.5359727373510239;   
  625    256          NaN    0.3412170149765343   0.5013584029891524;
  625    512          NaN    0.3205348495560217   0.4685884599523215;  
  #  
  1250    16   NaN    0.5922015791141663   0.5985338901911047;
  1250    64   NaN    0.44347375327882727   0.4931108428842008;
  1250    128  NaN    0.3643014387165608   0.4670902969568351;
  1250    256  NaN    0.29834512412292363   0.4487762628977663;
  1250    512   NaN   0.29592564642898533   0.40200031804656705;  
  #  
  2500    16   NaN        0.5846099784762775   0.5858367559405429;
  2500    64   NaN        0.41955791114224955   0.42975142901525815;
  2500    128  NaN        0.35361706523220293   0.39409471210456065;
  2500    256  NaN        0.2915633849235423   0.3884191981472285;
  2500    512   NaN       0.2958462986309559   0.3772805088697216;  
  #  
  5000    16    NaN  0.5642362907347902   0.5642137080294924;
  5000    64   NaN   0.39834187948318306   0.39955042032255117;
  5000    128  NaN   0.32619198020886664   0.34002659626785214;
  5000    256  NaN   0.2809545428074056   0.33578151805700684;
  5000    512  NaN   0.26318155375931673   0.34720586241623386;  
  #  
  10000    16  NaN  0.5335143273672013   0.5359159838750995;
  10000    64  NaN  0.3738634203327095   0.378029226143345;
  10000    128 NaN  0.3190183719909734   0.3282173392003402;
  10000    256 NaN  0.29051787628592013   0.316697875734246;
  10000    512 NaN  0.2768085267135688   0.3294943850118405;
  #  
  20000    16   NaN  0.5221457910699245   0.5225628739955206;
  20000    64  NaN   0.36743872889539037   0.3686846837880097;
  20000    128 NaN   0.30326480052171884   0.3069581188178217;
  20000    256 NaN   0.2837715236808136   0.30063817836691825;
  20000    512 NaN   0.29019720275307165   0.32058961207770803
]



PARA_Data = 
 [ 156    16         NaN        0.5563371083440583   10.343318411515451;
  156    64         NaN         0.48904159241299355   6.3004951521269925;
  156    128         NaN        0.45884473858303465   2.596137974485289;
  156    256        NaN         0.4350896524140923   1.5557021547221705;
  156    512        NaN         0.42147185719652397   1.524680620652429;
  #  
  312    16          NaN        0.5833138541639309   2.5442421774757475;
  312    64          NaN        0.4817353508989709   2.360320638900649;
  312    128         NaN        0.43368176104887185   1.5677485241449325;
  312    256         NaN        0.4152446417592739   1.2893834255540793;
  312    512         NaN        0.40743326472083463   1.1823912316330514;
  #  
  625    16           NaN       0.5960592617716124   1.1926520338673634;
  625    64           NaN       0.46194477736063816   1.383583653010105;   
  625    128         NaN        0.4120095857819295   1.1255411670007847;   
  625    256          NaN       0.3937622099034456   0.9636680992185122;
  625    512         NaN        0.3781724252962077   0.920034958683313;  
  #  
  1250    16    NaN      0.5674079421257973   0.8415755140049413;
  1250    64   NaN       0.45559920384495345   0.9723304302788807;
  1250    128  NaN       0.38024528157779214   0.8627939383499231;
  1250    256  NaN       0.3713497946153173   0.8244571816611195;
  1250    512   NaN      0.35041284958182023   0.7788598263919326;  
  #  
  2500    16   NaN    0.5027502458773784   0.5286808905168122;
  2500    64   NaN    0.4449561444821009   0.73451766044408771;
  2500    128  NaN    0.37626971896506356   0.7313744014482595;
  2500    256  NaN    0.34378949069728576   0.6894238395022512;
  2500    512   NaN   0.32884981875633684   0.6623233421713854;  
  #  
  5000    16    NaN  0.5037056199460105   0.5121384397627282;
  5000    64   NaN   0.4076288772327762   0.5154308112047226;
  5000    128  NaN   0.3528563972043536   0.5606902065414328;
  5000    256  NaN   0.31430977118080355   0.5641776746504772;
  5000    512  NaN   0.29665637742656314   0.5421138565602384;  
  #  
  10000    16  NaN    0.5063509762088984   0.5129714034200318;
  10000    64  NaN   0.36534866305321245   0.39415467405821736;
  10000    128 NaN   0.3271670929713664   0.41095134855871274;
  10000    256 NaN   0.2934986639145337   0.4478076835927357;
  10000    512 NaN   0.2584055401451376   0.4622577535768686;
  #
  20000    16  NaN   0.48832982758060334   0.4915810355668752;
  20000    64  NaN   0.33870733204043174   0.3458384420485145;
  20000    128 NaN   0.3077891623587318   0.3332230274374148;
  20000    256 NaN   0.27861453153064414   0.3449216239273876;
  20000    512 NaN   0.2598783272754618   0.38357742966324465;
]

FNO_Data =
 [ 156    2         NaN        0.43164076675207186   0.42791962298827296;
  156    4         NaN         0.3972085000803837   0.39999537360973847;
  156    8         NaN         0.3625229516854653   0.39046150422058046;
  156    16        NaN         0.2722322104546504   0.44352972072859603;
  156    32        NaN         0.15233223240535992   0.562338418016831;
  #  
  312    2          NaN        0.41764820796939045   0.4276630054108607;
  312    4          NaN        0.3763572618795129   0.39247389051776665;
  312    8         NaN         0.35681655281820357   0.39335009632393336;
  312    16         NaN        0.29058317207277584   0.41580210879254037;
  312    32         NaN        0.16038493670594806   0.533915412110778;
  #  
  625    2           NaN          0.4044492810726166   0.39591882445812226;
  625    4           NaN          0.3723292766928673   0.3664644246935844;   
  625    8         NaN            0.3463409193634987   0.35350839669704437;   
  625    16          NaN          0.2846962250471115   0.3854504640460014;
  625    32         NaN           0.17003533073961735   0.5104277313172817;  
  #  
  1250    2    NaN      0.42982933604717255   0.43239529428482054;
  1250     4   NaN      0.374323823428154   0.3793485984623432;
  1250     8  NaN       0.33523406203389167   0.34688754484057427;
  1250    16  NaN       0.30334076735675336   0.36033376272022727;
  1250    32   NaN      0.18673833864033224   0.49303058824241164;  
  #  
  2500    2   NaN        0.37873763887882234   0.38017223664820193;
  2500     4   NaN       0.34877212142944336   0.350284433093667;
  2500     8  NaN        0.3285831506967545   0.3341101096302271;
  2500    16  NaN        0.3076317037269473   0.33820216542631387;
  2500    32   NaN       0.18922108851522207   0.4829151016265154;  
  #  
  5000    2    NaN        0.3862793280258775   0.38404392939805987;
  5000    4   NaN         0.3478806491225958   0.34674211565703156;
  5000     8  NaN         0.32718553814291956   0.32999823293834923;
  5000    16  NaN         0.29816785604581236   0.3190270729139447;
  5000    32  NaN         0.24557640367746353   0.37487565384581684;  
  #  
  10000    2  NaN        0.3897208580002189   0.39186166196390987;
  10000    4  NaN        0.3447773130826652   0.3479163089081645;
  10000    8 NaN         0.3169546959228814   0.3214624141976237;
  10000    16 NaN        0.3005449684098363   0.3123433453690261;
  10000    32 NaN        0.2553581691533327   0.3509199070185423;
  #  
  20000    2   NaN       0.3636964907310903   0.3639958980873227;
  20000    4   NaN       0.3392967205397785   0.3396419344455004;
  20000    8   NaN       0.3252163030080497   0.32623238510079683;
  20000    16  NaN       0.298658544806391   0.30372749890722334;
  20000    32  NaN       0.27235044758245347   0.30749153174255045
]
######################################################







# nns = ["PCA-Net", "DeepO-Net", "PARA-Net", "FNO"]
# colors = ["C0","C1", "C2", "C3"]
# markers = ["o", "s", "^", "*"]
# linestyle = ["dotted", "-.", "--", "-", ]

# width

fig, ax = PyPlot.subplots(ncols = 4, sharex=false, sharey=true, figsize=(24,6))
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
plt.savefig("Advection-dc-Width-Error.pdf")




## Data vs Error
fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(24,6))
row_ids = [1,2,3,4]
# small

N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]

for i = 1:3  
    ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none")
    ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none")
    ax[i].loglog(N_Data, PARA_Data[row_ids[i]:5:40, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none")
    ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none")
    
    ax[i].loglog(N_Data, 10*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=1)
    ax[4].text(11000,1.4,"1/√N",color="#bababa",fontsize=22)
end

ax[4].loglog(N_Data, 10*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=1)
ax[4].text(11000,1.4,"1/√N",color="#bababa",fontsize=22)
ax[4].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",      label =  nns[1]  )
ax[4].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",      label =  nns[2]  )
ax[4].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",      label =  nns[3]  )
ax[4].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",      label =  nns[4]  )

# plot2_yticks = [0.1, 1]
for i = 1:4
    ax[i].set_title(sizes[i],pad=50)   
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].set_xticks(N_Data[2:2:end])
    ax[i].set_xticklabels(N_Data[2:2:end])
    # ax[i].set_yticks(plot2_yticks)

    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](which="minor",left=false) # remove minor ytick labels?
    ax[i].set_xlabel(latexstring("Training data ",L"N"),labelpad=20)
end
# ax[4].legend(frameon=false)
ax[1].set_ylabel("Test error")
# ax[1].set_yticklabels(plot2_yticks)
fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.87),ncol=4,frameon=false)

plt.tight_layout()
plt.savefig("Advection-dc-Data-Error.pdf")









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
    PARA_Data[i, 3] = PARA_Net_Cost(n_in, PARA_Data[i, 2],layers, Np)
end
for i = 1:size(FNO_Data)[1]
    FNO_Data[i, 3] = FNO_Net_Cost(FNO_Data[i, 2], kmax, layers, Np)
end

fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(24,6))


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
    ax[i].spines["bottom"].set_color("#808080")
    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](colors="#808080")
    ax[i].set_xticks([1e5, 1e7,1e9])
    ax[i].set_xticklabels([L"10^5",L"10^7",L"10^9"])
    ax[i].set_xlabel("Evaluation complexity",labelpad=20)
end
ax[4].legend(frameon=false,handlelength=0)
ax[1].set_ylabel("Test error")

plt.tight_layout()
plt.savefig("Advection-dc-Cost-Error.pdf")






