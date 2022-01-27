using LinearAlgebra
using PyPlot
include("../../nn/mynn.jl")
include("../../plotdefaults.jl")



# #  Data  | width | cost  |  Training-Time | Test-Error  |  
# PCA_Data = 
# [ 156    16         NaN    0.032267743729887055    229.08959536237387;
#   156    64         NaN    0.006354145223240747    28.603322319093593;
#   156    128         NaN   0.004091377279465882    7.339062927969896;
#   156    256        NaN    0.0023633952689072948   5.953647450911203;
#   156    512        NaN    0.0015557018631972956   0.7752412036121645;
#   #  
#   312    16          NaN   0.02851977728513595   44.32306211092087;
#   312    64          NaN   0.004451787035790968   1.3183049048134343;
#   312    128         NaN   0.0024728106076248356   0.5762955969395934;
#   312    256         NaN   0.0013561887200766187    0.7030645813625048;
#   312    512         NaN   0.0009216677074719118    0.3968664196788088;
#   #  
#   625    16           NaN    0.01735479654579431    0.19843738108281253;
#   625    64           NaN    0.003672739846786014    0.16018936485720067;  
#   625    128         NaN     0.0017351671613586035    0.15765189380959013;   
#   625    256          NaN    0.0009537160483335651    0.15275627589116839;
#   625    512         NaN     0.0007144833878313992    0.15286939018641785;  
#   #  
#   1250    16    NaN   0.012003287427061351    0.1368787324059245;
#   1250    64   NaN    0.0027691313950998966   0.10229087202917068;
#   1250    128  NaN    0.0014175980942963435    0.10275163682484068;
#   1250    256  NaN    0.0009093937215063658    0.10538724441906679;
#   1250    512   NaN   0.0009660852054757477  0.10291889470877152; 
#   #  
#   2500    16   NaN  0.014759171072238007    0.09322956872493414;
#   2500    64   NaN  0.002830806054013704    0.06398372711299864;
#   2500    128  NaN  0.001613800517601699    0.0643377555147;
#   2500    256  NaN  0.0013709734071609158   0.0654235179364261;
#   2500    512   NaN  0.0014014011507300212    0.06560974872756495;  
#   #  
#   5000    16    NaN   0.01736176197437385   0.04016330379145062;
#   5000    64   NaN    0.0032006592753933204    0.03680332249940026;
#   5000    128  NaN    0.002048918958871488    0.03836502640816897;
#   5000    256  NaN    0.0016615086952105433    0.039396281906188504;
#   5000    512  NaN    0.0012087616516309166    0.03840001187073026;  
#   #  
#   10000    16  NaN   0.015546498724072802   0.021775840093859026;
#   10000    64  NaN   0.0036814873455766543    0.021964745236877553;
#   10000    128 NaN   0.0021727671893241925    0.022485943700689053;
#   10000    256 NaN   0.0013476702312730392    0.021862497043584624;
#   10000    512 NaN   0.0009288358813897781    0.02157227483198821;
#   #  
#   20000    16   NaN 0.014009909556163252    0.016364888001558563;
#   20000    64  NaN  0.005036539922089507    0.013571462559800131;
#   20000    128 NaN 0.0024478788508812696    0.014473461109976318;
#   20000    256 NaN 0.0011644308861986463    0.014937867565655068;
#   20000    512 NaN  0.0008323537354247879   0.015102786765552685
# ]

# DeepONet_Data = 
#  #################################  
#  [ 156    16         NaN     0.04792660424935884    412.2184447190154;
#   156    64         NaN      0.009841323181509067    31.90274624001988;
#   156    128         NaN     0.005456871150169087    7.590282692541046;
#   156    256        NaN       0.0028961886519926613    1.530030504172064;
#   156    512        NaN       0.0024502575508603733    0.677267711676454;
#   #  
#   312    16          NaN     0.03601850543865765   61.61392180387418;
#   312    64          NaN     0.006507130530178613   1.4569146965867548;
#   312    128         NaN      0.003421115650458072   0.3503665429338388;
#   312    256         NaN      0.002173315657004154   0.3438373991778076;
#   312    512         NaN      0.00187543775647452   0.263914773071295;
#   #  
#   625    16           NaN   0.01933513785832105    0.2536387806786137;
#   625    64           NaN   0.004551824224823052    0.16574249572581817;   
#   625    128         NaN    0.002422441803964536    0.16000034754423303;   
#   625    256          NaN   0.0015418374449303054    0.15475618080282807;
#   625    512         NaN    0.001193587913190778    0.15445717579434423;  
#   #  
#   1250    16   NaN    0.016075554507949494    0.15550198646497634;
#   1250    64   NaN    0.0038518661575093916    0.11234480967418813;
#   1250    128  NaN    0.0020171094715565985    0.10568479820543648;
#   1250    256  NaN    0.0014011077238741177    0.10460073145933593;
#   1250    512   NaN   0.0012402545045734786    0.10017561041407462;  
#   #  
#   2500    16   NaN        0.016369408641842255    0.0901434305163784;
#   2500    64   NaN        0.003270696863155166    0.06654913652300898;
#   2500    128  NaN        0.002011565070224387    0.06410487954632661;
#   2500    256  NaN        0.0015533551228970716    0.0629279073338015;
#   2500    512   NaN       0.0014349536367269547    0.06301402397497774;  
#   #  
#   5000    16    NaN  0.01791614891840884    0.03843392765511875;
#   5000    64   NaN   0.0034485396028856685    0.03730765992301015;
#   5000    128  NaN   0.0020555829913873488    0.03630622436686346;
#   5000    256  NaN   0.0015316220075418507    0.03534459500317004;
#   5000    512  NaN   0.0013947171019329633    0.03620013529844387;  
#   #  
#   10000    16  NaN  0.017195191074751592    0.023514227347924202;
#   10000    64  NaN  0.0041874831903679485    0.022334643235098088;
#   10000    128 NaN  0.0020524571462924787    0.0204962661308532;
#   10000    256 NaN  0.0013822306700824686    0.02028674589130089;
#   10000    512 NaN  0.0010940795679694906    0.020517954281636566;
#   #  
#   20000    16   NaN  0.015558461934753381    0.017718287140232016;
#   20000    64  NaN   0.004780514236593877    0.013128297856002143;
#   20000    128 NaN   0.0023932891804109473    0.013641025627967391;
#   20000    256 NaN   0.0014939593680884525    0.013666943616649673;
#   20000    512 NaN   0.0010001303521358286    0.013634806898516203
# ]



# PARA_Data = 
#  [ 156    16         NaN        0.0898278219939592    27013.0267113816;
#   156    64         NaN         0.0419676400602999    152510.0938681436;
#   156    128         NaN        0.028978803488543568    147628.02457619173;
#   156    256        NaN         0.021092869073367722    22147.483696406525;
#   156    512        NaN         0.026784302558738    117074.70657572073;
#   #  
#   312    16          NaN        0.07634422533635325    11569.233099252402;
#   312    64          NaN        0.03245510354035484    29920.545932848818;
#   312    128         NaN        0.023433467146135558    19103.307465289494;
#   312    256         NaN        0.025017335495561694    2208.522421971045;
#   312    512         NaN        0.027304354536894813    10897.746903496627;
#   #  
#   625    16           NaN       0.07892960069682806    1.1261843631448638;
#   625    64           NaN       0.022858582235274708    0.2184609198426085;   
#   625    128         NaN        0.016644426624760712    0.15810098564883301;   
#   625    256          NaN       0.016225653663381162    0.2375947044816639;
#   625    512         NaN        0.017570326457580554    0.2785708674854613;  
#   #  
#   1250    16    NaN  0.045064634780811616    0.3957517192467675;
#   1250    64   NaN   0.015976055177280818    0.1224020902843938;
#   1250    128  NaN   0.019872092430852466    0.09195680065075552;
#   1250    256  NaN   0.010841210494105689    0.10300389359043348;
#   1250    512   NaN  0.01908727101628872    0.1257371793240066;  
#   #  
#   2500    16   NaN    0.0376753479950438    0.10795931677325789;
#   2500    64   NaN    0.006343319325890354    0.05676754929471231;
#   2500    128  NaN    0.010747092915305    0.04600084159871998;
#   2500    256  NaN    0.008319986050950249    0.048973786154427705;
#   2500    512   NaN   0.02988664096676383    0.06552418759517886;  
#   #  
#   5000    16    NaN  0.041837024779956065    0.045819434676902526;
#   5000    64   NaN   0.0060371628213841955    0.0294671324960315;
#   5000    128  NaN   0.014051280067606411    0.02279954818444857;
#   5000    256  NaN   0.0032357026307219157    0.01630637176583308;
#   5000    512  NaN   0.0032407626633889565    0.018708252177574727;  
#   #  
#   10000    16  NaN  0.0345032823842398    0.03668021167394363;
#   10000    64  NaN   0.006376293022078273   0.01566272310699992;
#   10000    128 NaN   0.005643324993156779   0.010099514347330465;
#   10000    256 NaN   0.004114225218143751   0.007540495229838777;
#   10000    512 NaN   0.005214902713069681   0.010653353165882692;
# ]

# FNO_Data =
#  [ 156    2         NaN        0.2210014521693572    0.23752738688236627;
#   156    4         NaN         0.09932527418893117    0.15493627513448396;
#   156    8         NaN         0.05872578216859928    0.10715191720578915;
#   156    16        NaN         0.017675954311226424    0.07304969742798652;
#   156    32        NaN         0.0049165971830893215    0.0644301198160228;
#   #  
#   312    2          NaN        0.19068203238436046    0.2061843552555029;
#   312    4          NaN        0.07601359707470505    0.10283708116278434;
#   312    8         NaN         0.04309874547358889    0.06855946381648;
#   312    16         NaN        0.013748772325925529    0.03994501317039323;
#   312    32         NaN        0.00357708932670693    0.02894336903969256;
#   #  
#   625    2           NaN          0.13797209955453874    0.14881067790985109;
#   625    4           NaN          0.052852927967906    0.0719697061240673;   
#   625    8         NaN            0.0031742428321391344    0.014855645129084587;   
#   625    16          NaN          0.010527498410642147    0.02570141346901655;
#   625    32         NaN           0.0031742428321391344    0.014855645129084587;  
#   #  
#   1250    2    NaN 0.12179827310442924    0.12589443739652634;
#   1250     4   NaN 0.04529312025755644    0.05522139312773943;
#   1250     8  NaN 0.020865313566476106    0.025714739104360344;
#   1250    16  NaN 0.007249637369811535    0.012232744136080146;
#   1250    32   NaN  0.002757601120043546    0.007683484995365143;  
#   #  
#   2500    2   NaN 0.09664646586477757    0.09715561604052782;
#   2500     4   NaN 0.03228148005604744    0.0355832889392972;
#   2500     8  NaN 0.015420772982388736    0.016956633882038295;
#   2500    16  NaN 0.0060910088106058535    0.008330654066614807;
#   2500    32   NaN  0.0025998110157437622    0.004628800611104816;  
#   #  
#   5000    2    NaN           0.0727172123402357    0.07298071169108153;
#   5000    4   NaN             0.02807683839481324    0.030169461058452727;
#   5000     8  NaN           0.011528103581815958    0.012406232164707034;
#   5000    16  NaN          0.004070917027094402    0.004966720711044036;
#   5000    32  NaN       0.0023880264116218314    0.003249396374542266;  
#   #  
#   10000    2  NaN        0.06032699879035354    0.061006014657393096;
#   10000    4  NaN          0.019344794865325095    0.01999789428981021;
#   10000    8 NaN       0.0087024106613826    0.009085109131829814;
#   10000    16 NaN        0.003177880827698391    0.003604570137267001;
#   10000    32 NaN         0.0017852196126536002    0.0022909274795325472;
#   #  
#   20000    2   NaN    0.050698157728835944    0.05094623484509066;
#   20000    4   NaN        0.017173789267451504    0.017578801312251015;
#   20000    8   NaN          0.006442254879069515    0.0066370841916068455;
#   20000    16  NaN              0.0024305332725256448    0.0026269240010529755;
#   20000    32  NaN        0.001703380346353515    0.001933683676365763
# ]
# ######################################################



#  Data  | width | cost  |  Training-Time | Test-Error  |  
PCA_Data = 
[ 156    16         NaN    0.032267743729887055    0.47889140015479154;
  156    64         NaN    0.006354145223240747    0.3239868632636588;
  156    128         NaN   0.004091377279465882    0.34646036163894955;
  156    256        NaN    0.0023633952689072948   0.28822991311095547;
  156    512        NaN    0.0015557018631972956   0.2586444672070481;
  #  
  312    16          NaN   0.02851977728513595      0.2783780838356531;
  312    64          NaN   0.004451787035790968     0.2155600958074504;
  312    128         NaN   0.0024728106076248356    0.1786010256867272;
  312    256         NaN   0.0013561887200766187    0.16258213055365092;
  312    512         NaN   0.0009216677074719118    0.14813191843108905;
  #  
  625    16           NaN    0.01735479654579431    0.17335641152723172;
  625    64           NaN    0.003672739846786014   0.1425600688383781;  
  625    128         NaN     0.0017351671613586035  0.11185524280882538;   
  625    256          NaN    0.0009537160483335651  0.09256800159936282;
  625    512         NaN     0.0007144833878313992  0.0860696150533358;  
  #  
  1250    16    NaN   0.012003287427061351    0.09130083906837172;
  1250    64   NaN    0.0027691313950998966   0.0911430230385822;
  1250    128  NaN    0.0014175980942963435   0.07387858799189574;
  1250    256  NaN    0.0009093937215063658   0.06377855252743928;
  1250    512   NaN   0.0009660852054757477   0.0604785786120294; 
  #  
  2500    16   NaN  0.014759171072238007    0.0699396439868313;
  2500    64   NaN  0.002830806054013704    0.0572728403983057;
  2500    128  NaN  0.001613800517601699    0.050673564700636345;
  2500    256  NaN  0.0013709734071609158   0.045130854339595065;
  2500    512   NaN  0.0014014011507300212  0.04459481957214915;  
  #  
  5000    16    NaN   0.01736176197437385     0.06079893076545126;
  5000    64   NaN    0.0032006592753933204   0.035590892947009255;
  5000    128  NaN    0.002048918958871488    0.034706487323358;
  5000    256  NaN    0.0016615086952105433   0.03395203569355669;
  5000    512  NaN    0.0012087616516309166   0.03423573460605307;  
  #  
  10000    16  NaN   0.015546498724072802    0.05612374521724525;
  10000    64  NaN   0.0036814873455766543   0.02737945016614654;
  10000    128 NaN   0.0021727671893241925   0.02273702561980498;
  10000    256 NaN   0.0013476702312730392   0.024810332277159966;
  10000    512 NaN   0.0009288358813897781   0.025339703776529474;
  #  
  20000    16   NaN 0.014009909556163252    0.0503812746451094;
  20000    64  NaN  0.005036539922089507    0.025027905599941856;
  20000    128 NaN 0.0024478788508812696    0.021431018929640038;
  20000    256 NaN 0.0011644308861986463    0.022192934294307555;
  20000    512 NaN  0.0008323537354247879   0.02227325281178602
]

DeepONet_Data = 
 #################################  
 [ 156    16         NaN     0.04792660424935884     0.4791780929981356;
  156    64         NaN      0.009841323181509067    0.40070574189362407;
  156    128         NaN     0.005456871150169087    0.38718573118742444;
  156    256        NaN       0.0028961886519926613  0.3938602965488228;
  156    512        NaN       0.0024502575508603733  0.4365021025354879;
  #  
  312    16          NaN     0.03601850543865765     0.4859861470559676;
  312    64          NaN     0.006507130530178613    0.3646712500219903;
  312    128         NaN      0.003421115650458072   0.33873165924847193;
  312    256         NaN      0.002173315657004154   0.33492114113941657;
  312    512         NaN      0.00187543775647452    0.29556141998861785;
  #  
  625    16           NaN   0.01933513785832105     0.46463221102662844;
  625    64           NaN   0.004551824224823052    0.29503896814145947;   
  625    128         NaN    0.002422441803964536    0.23809504362913422;   
  625    256          NaN   0.0015418374449303054   0.2651571526590951;
  625    512         NaN    0.001193587913190778    0.2528408220049244;  
  #  
  1250    16   NaN    0.016075554507949494    0.4677764495203692;
  1250    64   NaN    0.0038518661575093916    0.28676265821815544;
  1250    128  NaN    0.0020171094715565985   0.2222502132705782;
  1250    256  NaN    0.0014011077238741177   0.2209266842494748;
  1250    512   NaN   0.0012402545045734786   0.2009790703074071;  
  #  
  2500    16   NaN        0.016369408641842255    0.4691231500858085;
  2500    64   NaN        0.003270696863155166    0.2805182668599333;
  2500    128  NaN        0.002011565070224387    0.17913685907690993;
  2500    256  NaN        0.0015533551228970716   0.1836189103911095;
  2500    512   NaN       0.0014349536367269547   0.141294260611094;  
  #  
  5000    16    NaN  0.01791614891840884    0.46630838815102216;
  5000    64   NaN   0.0034485396028856685  0.20653094161615987;
  5000    128  NaN   0.0020555829913873488  0.145844974467108;
  5000    256  NaN   0.0015316220075418507  0.1437130818419545;
  5000    512  NaN   0.0013947171019329633  0.1405473816820136;  
  #  
  10000    16  NaN  0.017195191074751592    0.4655734764893873;
  10000    64  NaN  0.0041874831903679485    0.16514735215927848;
  10000    128 NaN  0.0020524571462924787    0.14300845009868007;
  10000    256 NaN  0.0013822306700824686    0.11138742740606519;
  10000    512 NaN  0.0010940795679694906    0.10635394060840886;
  #  
  20000    16   NaN  0.015558461934753381    0.4668837018826419;
  20000    64  NaN   0.004780514236593877    0.154620569555557;
  20000    128 NaN   0.0023932891804109473   0.10774769727794059;
  20000    256 NaN   0.0014939593680884525   0.0899485789369525;
  20000    512 NaN   0.0010001303521358286   0.06991753984867992
]



PARA_Data = 
 [ 156    16         NaN     0.04792660424935884     0.995351067459419;
  156    64         NaN      0.009841323181509067    1.8456926207553455;
  156    128         NaN     0.005456871150169087    2.552705497352204;
  156    256        NaN       0.0028961886519926613  2.4053246120486484;
  156    512        NaN       0.0024502575508603733  3.263985956107103;
  #  
  312    16          NaN     0.03601850543865765     0.6874816539455738;
  312    64          NaN     0.006507130530178613    0.5766836045852393;
  312    128         NaN      0.003421115650458072   0.8619637412307173;
  312    256         NaN      0.002173315657004154   0.6899915855873809;
  312    512         NaN      0.00187543775647452    0.6540565254096767;
  #  
  625    16           NaN   0.01933513785832105     0.5447519148799588;
  625    64           NaN   0.004551824224823052    0.28019432222501767;   
  625    128         NaN    0.002422441803964536    0.3079983966454312;   
  625    256          NaN   0.0015418374449303054   0.27680360629262396;
  625    512         NaN    0.001193587913190778    0.2528408220049244;  
  #  
  1250    16   NaN    0.016075554507949494    0.5574791778724401;
  1250    64   NaN    0.0038518661575093916   0.24614157219869567;
  1250    128  NaN    0.0020171094715565985   0.19409349553361316;
  1250    256  NaN    0.0014011077238741177   0.17345010529617372;
  1250    512   NaN   0.0012402545045734786   0.1475752402030587;  
  #  
  2500    16   NaN        0.016369408641842255    0.4855703755280405;
  2500    64   NaN        0.003270696863155166    0.22254220909267014;
  2500    128  NaN        0.002011565070224387    0.1851885594222058;
  2500    256  NaN        0.0015533551228970716   0.14803208714850774;
  2500    512   NaN       0.0014349536367269547   0.14021073329010636;  
  #  
  5000    16    NaN  0.01791614891840884    0.5024158045240535;
  5000    64   NaN   0.0034485396028856685  0.2439310361373483;
  5000    128  NaN   0.0020555829913873488  0.16867432960104953;
  5000    256  NaN   0.0015316220075418507  0.15316442317846823;
  5000    512  NaN   0.0013947171019329633  0.11833008403500189;  
  #  
  10000    16  NaN  0.017195191074751592    NaN;
  10000    64  NaN  0.0041874831903679485    NaN;
  10000    128 NaN  0.0020524571462924787    NaN;
  10000    256 NaN  0.0013822306700824686    NaN;
  10000    512 NaN  0.0010940795679694906    NaN;
]

FNO_Data =
 [ 156    2         NaN        0.2210014521693572    0.16391316591165003;
  156    4         NaN         0.09932527418893117    0.12510674542341477;
  156    8         NaN         0.05872578216859928    0.11014038856881551;
  156    16        NaN         0.017675954311226424    0.10516279408087333;
  156    32        NaN         0.0049165971830893215    0.10707662502924602;
  #  
  312    2          NaN        0.19068203238436046    0.13845041679600492;
  312    4          NaN        0.07601359707470505    0.1053805681041036;
  312    8         NaN         0.04309874547358889    0.0822056943598466;
  312    16         NaN        0.013748772325925529    0.07849590218840884;
  312    32         NaN        0.00357708932670693    0.0828954488850939;
  #  
  625    2           NaN          0.13797209955453874    0.1184616432607174;
  625    4           NaN          0.052852927967906        0.08209335761666298;   
  625    8         NaN            0.0031742428321391344    0.059179240342974664;   
  625    16          NaN          0.010527498410642147    0.05142759363949299;
  625    32         NaN           0.0031742428321391344    0.05867505546808243;  
  #  
  1250    2    NaN 0.12179827310442924    0.10422199742496013;
  1250     4   NaN 0.04529312025755644    0.06693224086761475;
  1250     8  NaN 0.020865313566476106    0.043567280170321465;
  1250    16  NaN 0.007249637369811535    0.034244843447208406;
  1250    32   NaN  0.002757601120043546  0.03811766854301095;  
  #  
  2500    2   NaN 0.09664646586477757    0.09162359144985675;
  2500     4   NaN 0.03228148005604744   0.05823724198192358;
  2500     8  NaN 0.015420772982388736   0.03406920659095049;
  2500    16  NaN 0.0060910088106058535   0.022838749076612295;
  2500    32   NaN  0.0025998110157437622    0.022813285333290696;  
  #  
  5000    2    NaN           0.0727172123402357    0.08231497655063867;
  5000    4   NaN             0.02807683839481324  0.04945031707957387;
  5000     8  NaN           0.011528103581815958    0.026391304425708948;
  5000    16  NaN          0.004070917027094402    0.015542736510187387;
  5000    32  NaN       0.0023880264116218314     0.01224069413091056;  
  #  
  10000    2  NaN        0.06032699879035354       0.07463171821348369;
  10000    4  NaN          0.019344794865325095    0.04110900370404124;
  10000    8 NaN       0.0087024106613826           0.019516954693803563;
  10000    16 NaN        0.003177880827698391       0.011036665086820722;
  10000    32 NaN         0.0017852196126536002     0.007529913164558821;
  #  
  20000    2   NaN    0.050698157728835944               0.0687543995609507;
  20000    4   NaN        0.017173789267451504           0.035979969158163295;
  20000    8   NaN          0.006442254879069515         0.01583976535690017;
  20000    16  NaN              0.0024305332725256448    0.0082029201834579;
  20000    32  NaN        0.001703380346353515           0.0056016913049272265
]
######################################################



# width

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
ax[1].set_xlabel("Network width "*L"w",labelpad=20)
ax[2].set_xticks(DeepONet_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[2].set_xlabel("Network width "*L"w",labelpad=20)
ax[3].set_xticks(PARA_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[3].set_xlabel("Network width "*L"w",labelpad=20)
ax[4].set_xticks(FNO_Data[(i+3)*5+1:(i+3)*5+5, 2])
ax[4].set_xlabel("Lifting dimension "*L"d_f",labelpad=20)

plt.tight_layout()
plt.savefig("Helmholtz-Width-Error.pdf")




## Data vs Error
fig, ax = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(24,6))

row_ids = [1,2,3,4]
# small

N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]
for i = 1:3
    ax[i].loglog(N_Data, PCA_Data[row_ids[i]:5:40, 5],      color = colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none")
    ax[i].loglog(N_Data, DeepONet_Data[row_ids[i]:5:40, 5], color = colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none")
    ax[i].loglog(N_Data[1:7], PARA_Data[row_ids[i]:5:35, 5],     color = colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none")
    ax[i].loglog(N_Data, FNO_Data[row_ids[i]:5:40, 5],      color = colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none")
    
    ax[i].loglog(N_Data, 5*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",  linewidth=1)
    ax[i].text(10000,1,"1/√N",color="#bababa",fontsize=22)
end
i = 4
ax[4].loglog(N_Data, 5*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",linewidth=1)
ax[4].text(10000,1,"1/√N",color="#bababa",fontsize=22)
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
    # ax[i].set_yticks(plot2_yticks)

    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](which="minor",left=false) # remove minor ytick labels?
    ax[i].set_xlabel(latexstring("Training data ",L"N"),labelpad=20)
end
# ax[1].set_yticklabels(plot2_yticks)
ax[1].set_ylabel("Test error")

fig.legend(loc = "upper center",bbox_to_anchor=(0.5,0.87),ncol=4,frameon=false)
plt.tight_layout()
plt.savefig("Helmholtz-Data-Error.pdf")









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
 ax[i].title.set_text("N = "*string(Int(FNO_Data[(i+3)*5+1, 1])))   
end

for i = 1:4
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color("#808080")
    ax[i].spines["bottom"].set_color("#808080")
    ax[i][:xaxis][:set_tick_params](colors="#808080")
    ax[i][:yaxis][:set_tick_params](colors="#808080")
    # ax[i].set_xticks([1e5, 1e7,1e9])
    # ax[i].set_xticklabels([L"10^5",L"10^7",L"10^9"])
    ax[i].set_xlabel("Evaluation complexity",labelpad=20)
end
ax[1].legend(frameon=false,handlelength=0)
ax[1].set_ylabel("Test error")

plt.tight_layout()
plt.savefig("Helmholtz-Cost-Error.pdf")






