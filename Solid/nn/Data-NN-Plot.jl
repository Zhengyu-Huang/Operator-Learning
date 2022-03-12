using LinearAlgebra
using PyPlot
using LaTeXStrings
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
[ 156    16         NaN    0.11357795490419946   0.17300330266093625;
  156    64         NaN    0.04785650604632496   0.1555171438097476;
  156    128         NaN   0.027536000950855122   0.1579223654996282;
  156    256        NaN    0.015768088568243764   0.14859757432301784;
  156    512        NaN    0.008026710380706303   0.1458746046893197;
  #  
  312    16          NaN   0.11162090081002625   0.1332856119927404;
  312    64          NaN   0.04188688194109087   0.1259850142107492;
  312    128         NaN   0.02505676992697201   0.11702500944052442;
  312    256         NaN   0.012966973813441043   0.11168882987152419;
  312    512         NaN   0.006281242717713232   0.11088791873659272;
  #  
  625    16           NaN    0.08694705305316157   0.09876099162882608;
  625    64           NaN    0.041527379397561216   0.09367221797535263;  
  625    128         NaN     0.021951621638207505   0.09447500807075779;   
  625    256          NaN    0.011416432893756585   0.0903977865257086;
  625    512         NaN     0.005548063354824352   0.08796312961820399;  
  #  
  1250    16    NaN   0.08323959655787871   0.08807251119211822;
  1250    64   NaN    0.0423289096188587   0.07397797829919026;
  1250    128  NaN    0.02375796420488976   0.07633164395408479;
  1250    256  NaN    0.01050288738441434   0.07367534794701522;
  1250    512   NaN   0.005363692473632316   0.0709425149109577; 
  #  
  2500    16   NaN  0.07978360996581975   0.08121453080022611;
  2500    64   NaN  0.04431792431430159   0.05986327061637096;
  2500    128  NaN  0.0258367759724788   0.06715272160319648;
  2500    256  NaN  0.012044078904295343   0.0678187783988049;
  2500    512   NaN  0.006234549017533784   0.06433617711973359;  
  #  
  5000    16    NaN   0.07247650591487838   0.07333547620274454;
  5000    64   NaN    0.04496860444322229   0.05373009715231799;
  5000    128  NaN    0.029848955279902886   0.061380616373617115;
  5000    256  NaN    0.014633140447374331   0.06562974057643944;
  5000    512  NaN    0.009176110183490354   0.06345607667213714;  
  #  
  10000    16  NaN   0.07061621703634144   0.07085489891527146;
  10000    64  NaN   0.04644917374013051   0.04975507772933423;
  10000    128 NaN   0.03410576725858564   0.05636758222656388;
  10000    256 NaN   0.020109419310930302   0.06465998120322289;
  10000    512 NaN   0.012510275552019681   0.06519014232966086;
  #  
  20000    16   NaN 0.06968902028945718   0.07010400380030836;
  20000    64  NaN  0.04517700137929292   0.046707209402072826;
  20000    128 NaN  0.03722111748615488   0.051998938594154616;
  20000    256 NaN  0.025428953740643957   0.0624401348933131;
  20000    512 NaN  0.01677502076551009   0.06612517934417321
]

DeepONet_Data = 
 #################################  
 [ 156    16         NaN     0.14262153473786143   0.20911700972455477;
  156    64         NaN      0.0825461629468253   0.19916936629656584;
  156    128         NaN     0.0641641532454597   0.16563255786773382;
  156    256        NaN      0.044255925081537216   0.15127393976307635;
  156    512        NaN      0.03811256536137279   0.14573781939613803;
  #  
  312    16          NaN     0.14343064570456243   0.16446094056008415;
  312    64          NaN     0.07178047815061973   0.14051986776538725;
  312    128         NaN     0.0501239646953582   0.12588693831756098;
  312    256         NaN     0.03367887659480562   0.12146449232378018;
  312    512         NaN     0.033550645716084626   0.11555347780650226;
  #  
  625    16           NaN   0.1333004665686784   0.14259541545274743;
  625    64           NaN   0.06316670217364806   0.1111525155887363;   
  625    128         NaN    0.04136898984351309   0.10405714732813212;   
  625    256          NaN   0.024995251338080646   0.10021888340633221;
  625    512         NaN    0.02456987712904521   0.09220676893161515;  
  #  
  1250    16   NaN    0.11216805347881273   0.11420346606112022;
  1250    64   NaN    0.05779067057862701   0.08695544034015643;
  1250    128  NaN    0.03455626561063574   0.08472193283425433;
  1250    256  NaN    0.020715468874403977   0.07933179702239943;
  1250    512   NaN   0.017953511095994305   0.07584540043217332;  
  #  
  2500    16   NaN       0.10080369414676142   0.10171799749855132;
  2500    64   NaN       0.054940701752072246   0.07135116663367723;
  2500    128  NaN       0.03201521595577919   0.07287209885152629;
  2500    256  NaN       0.018432590252114155   0.07110235271260323;
  2500    512   NaN      0.015496711897852357   0.06732758228280428;  
  #  
  5000    16    NaN  0.10067461807061681   0.10119316827304131;
  5000    64   NaN   0.052689070700674814   0.06161910728686778;
  5000    128  NaN   0.034052042605518966   0.06608319049336267;
  5000    256  NaN   0.021703010813491578   0.06740272601048589;
  5000    512  NaN   0.017313206604374368   0.06580866265873055;  
  #  
  10000    16  NaN  0.09629186957432598   0.09650476270208726;
  10000    64  NaN  0.051094177404038627   0.05535873675459766;
  10000    128 NaN  0.03697136997184957   0.0595448008144785;
  10000    256 NaN  0.02800886536647485   0.06303770903394633;
  10000    512 NaN  0.020595466556320857   0.06386476327391244;
  #  
  20000    16   NaN  0.08214447922175039   0.0824746149540089;
  20000    64  NaN   0.05029818244161458   0.052014380370673234;
  20000    128 NaN   0.04050047397807624   0.052232461655715316;
  20000    256 NaN   0.03114209951595663   0.05872446339965451;
  20000    512 NaN   0.02250886706988622   0.06377522281721379
]



PARA_Data = 
 [ 156    16         NaN        0.12126418187078628   0.155901216751596;
  156    64         NaN         0.08245048846883009   0.41591145950502917;
  156    128         NaN        0.061419083109788955   0.357271144783621;
  156    256        NaN         0.04460611304643863   0.24642358596844752;
  156    512        NaN         0.020462592671517718   0.2068158117051065;
  #  
  312    16          NaN        0.10902622193760088   0.11630362419063865;
  312    64          NaN        0.06890630506550056   0.14513550583748028;
  312    128         NaN        0.040747435441389986   0.15815065218910362;
  312    256         NaN        0.02887442987046901   0.14906296011183218;
  312    512         NaN        0.019619023615749357   0.13566762377560024;
  #  
  625    16           NaN       0.10877152930806612   0.11374351053573027;
  625    64           NaN       0.05386946046334069   0.07640636277074815;   
  625    128         NaN        0.03547439489346798   0.086732827963942;   
  625    256          NaN       0.019939377996368832   0.08422451133747831;
  625    512         NaN        0.013935312875606178   0.0786522827568006;  
  #  
  1250    16    NaN  0.1059231602704915   0.10758581411965022;
  1250    64   NaN   0.05595115201061856   0.06360704853494817;
  1250    128  NaN   0.034699818675691885   0.07040471408194512;
  1250    256  NaN   0.020329952513192077   0.06977129066766359;
  1250    512   NaN  0.012941735384026452   0.06481673088508715;  
  #  
  2500    16   NaN    0.10952269543325531   0.11075322470323072;
  2500    64   NaN    0.05273241326160547   0.05503011314962456;
  2500    128  NaN    0.037449122417834255   0.06231787379763691;
  2500    256  NaN    0.02279858533363507   0.0659675066794221;
  2500    512   NaN   0.014909880984090535   0.06216764057583373;  
  #  
  5000    16    NaN  0.10158911921752072   0.10191300548088568;
  5000    64   NaN   0.05102815179424756   0.05232287974647775;
  5000    128  NaN   0.04164499789361079   0.052333591220053904;
  5000    256  NaN   0.02618442197134807   0.06487806269774568;
  5000    512  NaN   0.017672231974254434   0.06327365592155092;  
  #  
  10000    16  NaN  0.09997978736445122   0.10006135662523578;
  10000    64  NaN  0.050298250430750265   0.05088164263264469;
  10000    128 NaN  0.04377474991972547   0.04694940425733192;
  10000    256 NaN  0.03012397095489149   0.060947419897955546;
  10000    512 NaN  0.023003220909326504   0.06427785429205589;
  #  
  20000    16  NaN  0.09582615248576822   0.09617832666157253;
  20000    64  NaN  0.05170648529604365   0.05213040457095706;
  20000    128 NaN  0.044387002190457596   0.04550915230284199;
  20000    256 NaN  0.03588759048996521   0.055779106939107204;  #todo
  20000    512 NaN  0.025725793707126184  0.06773422219860994; #todo
]

FNO_Data =
 [ 156    2         NaN        0.12054922604226757   0.14414993208999427;
  156    4         NaN         0.07358726692978793   0.09196700798726637;
  156    8         NaN         0.052437771172048425  0.07699770120775665;
  156    16        NaN         0.03416816555035162   0.07293769981143623;
  156    32        NaN         0.021201692385779125   0.08198683623096398;
  #  
  312    2          NaN        0.09380573048340005   0.11797815550410558;
  312    4          NaN        0.06780511970471245   0.07711609338557911;
  312    8         NaN         0.051782648274710394   0.06857374769010059;
  312    16         NaN        0.037710728324064004   0.06252741869388002;
  312    32         NaN        0.021732976140481374   0.07419626626244212;
  #  
  625    2           NaN       0.07907828340249898   0.09873100314521893;
  625    4           NaN       0.06243120087779477   0.068495422382559;   
  625    8         NaN         0.0512463722849898   0.0605111064388268;   
  625    16          NaN       0.04014251539210613   0.05735260729174315;
  625    32         NaN        0.022712888805342352   0.06894363440844378;  
  #  
  1250    2    NaN   0.0721604658552225   0.08559472825922429;
  1250     4   NaN   0.058587717519323075  0.06374132716451333;
  1250     8  NaN    0.05029862300579784   0.05644247173509566;
  1250    16  NaN    0.04331249470150343   0.05277286869792904;
  1250    32   NaN   0.0234019228670352    0.06401451171950027;  
  #  
  2500    2   NaN  0.06690852589616007   0.07703702018266063;
  2500     4  NaN  0.05543196399908585   0.05970591035402763;
  2500     8  NaN  0.049561563059286405  0.05269055695228226;
  2500    16  NaN  0.04600680480065976   0.049378732965302194;
  2500    32   NaN 0.026083269424247612  0.057459223701744284;  
  #  
  5000    2    NaN  0.06350585829725314   0.07335178690050474;
  5000    4   NaN   0.053625572579771665  0.05734815838413223;
  5000     8  NaN   0.04870559853052172   0.05149887586882659;
  5000    16  NaN   0.04598098710471801   0.04908295024692129;
  5000    32  NaN    0.031091677720447502   0.052560135418266235;  
  #  
  10000    2  NaN   0.06100410239285384   0.06987006746540537;
  10000    4  NaN   0.05150264963251262   0.055270600865062726;
  10000    8 NaN    0.04842893279212205   0.050226572697772816;
  10000    16 NaN   0.04668515263495779   0.048233262082042296;
  10000    32 NaN   0.043760100073658985   0.04803120092902064;
  #  
  20000    2   NaN    0.05900645581286424   0.06714351954759644;
  20000    4   NaN    0.05034109099491383   0.053047794020236856;
  20000    8   NaN    0.04785601547992645   0.04970943873454443;
  20000    16  NaN    0.04670982972818465   0.04777946294882538;
  20000    32  NaN    0.04619993448264109   0.047429043266354005
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

plt.subplots_adjust(bottom=0.25,top=0.85,left=0.08,right=0.98)
plt.savefig("Solid-Width-Error.pdf")
plt.close()

## Data vs Error
fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=nn_linefigsize)
plot2_yticks = [0.05, 0.1, 0.2, 0.4]

row_ids = [1,2,3,4]
# small

N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]
for i = 1:3
    ax[i].loglog(N_Data, 0.4*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=0.5)
    ax[i].text(300,0.32,"1/√N",color="#bababa",fontsize=7)
    ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none")
    ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none")
    ax[i].loglog(N_Data, PARA_Data[row_ids[i]:5:40, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none")
    ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none")
end

i = 4
ax[i].loglog(N_Data, 0.4*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=0.5)
ax[i].text(300,0.32,"1/√N",color="#bababa",fontsize=7)
ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",      label =  nns[1]  )
ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",      label =  nns[2]  )
ax[i].loglog(N_Data, PARA_Data[row_ids[i]:5:40, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",      label =  nns[3]  )
ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",      label =  nns[4]  )


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
    ax[i].set_yticks(plot2_yticks)

    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](which="minor",left=false) # remove minor ytick labels?
    ax[i].set_xlabel(latexstring("Training data ",L"N"),labelpad=2)
end
# ax[4].legend(frameon=false)
# fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.87),ncol=4,frameon=false)
ax[1].set_ylabel("Test error")
ax[1].set_yticklabels(plot2_yticks)

fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.92),ncol=4,frameon=false,fontsize=8)
plt.subplots_adjust(bottom=0.2,top=0.8,left=0.08,right=0.98)
plt.savefig("Solid-Data-Error.pdf")
plt.close()




# complexity
Np=41*41
layers = 4
n_in = 21
kmax = 12*12
for i = 1:size(PCA_Data)[1]
    PCA_Data[i, 3] = PCA_Net_Cost(n_in, PCA_Data[i, 2],layers, Np)
end
for i = 1:size(DeepONet_Data)[1]
    DeepONet_Data[i, 3] = DeepO_Net_Cost(n_in, DeepONet_Data[i, 2],layers, Np)
end
for i = 1:size(PARA_Data)[1]
    PARA_Data[i, 3] = PARA_Net_Cost(n_in, 2, PARA_Data[i, 2],layers, Np)
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
    ax[i].title.set_text(L"N = "*string(Int(FNO_Data[(i+3)*5+1, 1])))   
end

for i = 1:4
    # ax[i].title.set_text(nns[i])   
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color("#808080")
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false)
    ax[i].set_xticks([1e5, 1e7,1e9])
    ax[i].set_xticklabels([L"10^5",L"10^7",L"10^9"])
    ax[i].set_xlabel("Evaluation cost",labelpad=2)
end
ax[4].legend(frameon=false,handlelength=0)
ax[1].set_ylabel("Test error")


plt.subplots_adjust(bottom=0.22,top=0.85,left=0.08,right=0.98)
plt.savefig("Solid-Cost-Error.pdf")
plt.close()






