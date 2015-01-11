var dataMap = {};
function dataFormatter(obj) {
    var pList = ['北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江','上海','江苏','浙江','安徽','福建','江西','山东','河南','湖北','湖南','广东','广西','海南','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆'];
    var temp;
    var max = 0;
    for (var year = 2007; year <= 2014; year++) {
        temp = obj[year];
        for (var i = 0, l = temp.length; i < l; i++) {
            max = Math.max(max, temp[i]);
            obj[year][i] = {
                name : pList[i],
                value : temp[i]
            }
        }
        obj[year+'max'] = Math.floor(max/100) * 100;
    }
    return obj;
}

function dataMix(list) {
    var mixData = {};
    for (var i = 0, l = list.length; i < l; i++) {
        for (var key in list[i]) {
            if (list[i][key] instanceof Array) {
                mixData[key] = mixData[key] || [];
                for (var j = 0, k = list[i][key].length; j < k; j++) {
                    mixData[key][j] = mixData[key][j] 
                                      || {name : list[i][key][j].name, value : []};
                    mixData[key][j].value.push(list[i][key][j].value);
                }
            }
        }
    }
    return mixData;
}

dataMap.dataGDP = dataFormatter({
    //Return
    2007:[0.146163215590742,-0.228091236494597,0.253517980790708,-0.00108920596884927,-0.0152505446623094,-0.0370936068077682,-0.0219010074463413,0.0164618086040377,-0.157773638654541,-0.310586663698096,0.14016544117647,0.055517788352594,-0.294084507042253,0.337822149988391,-0.0561482313307131,-0.0937323546019196,-0.256497948016416,0.0982800982800987,-0.179585216081566,-0.219443133553563,-0.0844390832328109,-0.0839416058394158,0.00122684333210537,-0.0797350343473984,0.490911339186348,-0.122583686940123,0.221957040572792,-0.07121176745272,0.0493827160493829,0.83070083070083,0.478556767851356,-0.123711340206186,-0.123173277661795,0.262101035721834,-0.221421215242018,-0.619273301737758,-0.0639946109801272,-0.19322033898305,-0.125590505818643,0.116686114352392,-0.0519031141868515,-0.100869565217392,0.0655891309440153,-0.0453804980218758,0.0210403272939809,-0.115478828881373,0.00118008024545729,-0.0955752212389383,0.0250178699070774,0.23767082590612,0.00348229831688928,-0.337665351589697,0.193346943677195,-0.021206409048067,0.0708382526564338,-0.0105509964830016,-0.0445957047294913,0.126134622185548,0.384167636786961,0.0347533632286998,-0.0491565188247137],
    2008:[0.147881930124316,0.163144449605202,-0.0628391888032004,0.347801092267894,0.0522222222222221,0.209461700011053,0.104476803984194,-0.218043501553627,0.135275754422477,-0.0869988111963695,0.256745707277188,0.0956632653061231,-0.194777847967994,0.186298722216257,-0.111210667791071,-0.214262871762073,0.125272331154685,-0.4147391070468,-0.0639766541332278,0.232702626376728,0.290335044433406,0.0311108727136182,-0.0171113844179452,0.110343349938401,-0.140397350993378,-0.0392262224610419,-0.00971030911150708,0.228954047194773,-0.200073905928312,-0.217086834733892,0.221904080171795,-0.260180995475112,-0.417565400143798,-0.0513678864134818,0.259325868770668,0.259556661388826,-0.147715372319903,0.120279720279721,-0.311221669430624,-0.120385690648714,0.00519750519750539,0.238961038961038,-0.514685156998703,0.108760251990967,-0.155799870656711,0.432393693263259,-0.372108999313029,0.115352598406469,0.29626146249706,0.250627997259649,-0.295182400445559,0.136585365853658,-0.22929287210553,0.075327384401436,-0.243846330802853,0.186866305116719,-0.057867021584399,-0.387637506547931,0.0690281562216159,-0.256780323531179,0.26169608690285],
    2009:[0.0669895076674736,-0.0665437344664474,0.100080710250201,0.0551382451654146,0.112055948501947,0.379597610814209,0.0484591504505198,-0.01582397709291,-0.259622641509434,0.0100728343406164,0.0294140413344683,-0.39746854981863,-0.396238546857419,0.289563980249394,-0.0431069540463604,0.345531775853619,0.0631662060797471,-0.123970184386034,-0.134265512036228,-0.135287485907554,0.675918367346939,0.173573940969568,0.151822623074031,0.0547864070481979,0.260658272586701,0.0100466451381424,0.104667001218724,0.197942532813056,0.064700153054124,-0.056680721642358,-0.0785540493569688,-0.173066143497758,-0.0213903743315516,-0.212933190425151,-0.0642476454698106,0.0191050040414425,-0.0564723138980551,0.022127157397845,0.264939652634677,-0.151276168626325,-0.244594889713912,0.164913066189091,0.267214799588901,0.184470184470183,-0.0329963493400729,0.0324012115235619,0.0280839710735102,-0.19673738010222,-0.100699900014283,-0.231585022725634,0.134416543574593,-0.0626730797259864,0.158404224112643,0.275772451631533,-0.00491780244484988,0.324031770577071,0.0435729847494563,0.286740780911062,0.0764415156507411,-0.091557125106272,0.345191736519043],
    2010:[-0.0491082967174975,-0.276808905380333,-0.0167913295680048,0.269113149847093,-0.281045265038714,0.201463097016356,-0.28796696076591,-0.103989485078089,-0.380483612641119,-0.421928934010152,0.768676333418129,0.0996102208748365,0.217137065336036,-0.142317524514478,-0.175723796253291,0.015759199432669,-0.0731649752183143,-0.159296243461722,-0.426028831440767,0.191790040376851,0.183228788379003,-0.0624088182849729,-0.045265475899192,0.378517881283027,0.139332938622459,0.153768296480847,0.119618142084883,-0.0314453494980104,-0.272119185162663,-0.196124394436631,-0.0641587630509279,-0.245858901856978,0.300575657894737,0.119755698375314,0.0303735552838156,0.212757590058203,0.291116331010051,0.172871842843779,0.0809210284327064,-0.142299412558835,0.136210534108154,-0.105167062260361,-0.0727017751042561,-0.0855018587360599,0.0599925009373837,-0.452105851658591,-0.181129718546278,-0.121257901641952,-0.0619768190598838,0.0684376771685429,0.403812894662753,-0.0224223914640291,0.0592816459374648,-0.0897465526538781,-0.213377900423647,0.0369340746624296,-0.0506469354647252,-0.061642473652813,-0.172869147659063,0.256942747780763,0.0932946921275169],
    2011:[0.481427449220066,0.069057815845824,-0.0653942261683231,0.263833886332012,0.166588456123887,0.000512859963586476,-0.132311084899613,-0.0987449003456074,-0.183187675511114,0.152118700708461,0.110338652762416,0.00546974708931262,-0.173374290623209,0.0651690155769837,0.123177343791125,0.121678539857522,0.0940148985358327,0.191877035830619,0.277902621722846,0.0442144644462258,-0.0316846051517717,-0.250400349395836,0.0617191777412776,-0.0279495424189957,-0.0969814222288371,-0.056354255372439,-0.162212538726984,-0.236833346135136,-0.176229938109723,-0.0560597971169253,0.154362416107383,-0.0232650363516192,-0.200598881734107,0.514061654948621,0.295259895578818,0.0474655874490988,0.155150550734728,0.332247876013025,-0.0476303317535543,0.0535727041120027,-0.559397499052672,-0.0830364758416538,-0.061723710505679,0.328352890269045,-0.197156024348769,0.0711466425321176,0.102096854717922,0.00642467073562457,-0.00419804914186976,-0.204308513266465,0.0226980404025128,0.142422183639062,-0.0702111295804695,-0.0127423545872474,0.162609761589073,-0.269551731777566,-0.254503137016797,-0.0155755152899648,-0.139358327700068,0.252333491536149,-0.104415811537176],
    2012:[-0.0646735215141798,0.0161500304873027,0.130143635137138,0.111255481565696,-0.0263432656011563,-0.0169106633811656,0.014035427354564,0.0156267620382457,0.133504905903168,0.0268253968253977,-0.0136142728236954,0.0873452435679976,0.184964012949052,0.262918331764669,-0.136660903555589,0.195256531414811,-0.0932916710272551,0.00890391319439252,0.187871294593045,-0.0130239166469334,-0.0197095435684637,-0.142545324958796,0.0206365704128821,0.146263002826048,-0.0702253433485934,0.0901183175924677,0.0616617623138698,-0.260125800952326,-0.03244107794912,0.139269440954297,0.196927395153706,0.121526252598904,0.122960305520194,0.0304381377004201,0.00270689974498233,-0.048426150121065,0.0198940890224701,-0.132840063420418,-0.249714095455927,-0.124120319505894,0.242641089629876,-0.208712499266131,-0.115574876330386,0.0291178210163933,0.153332022803225,-0.0692531089433329,-0.213107378524294,-0.220965690556093,-0.0363539495745638,0.0795785169458197,-0.199872056919067,0.0256328610093936,0.0801956456146473,0.236782405948705,-0.0797180627587367,-0.185693675245498,-0.360394537177541,0.396661419388691,-0.326009810261343,0.056573627233599,-0.118184913185158],
    2013:[0.123495388463342,-0.0174269831465636,0.0917066271849388,0.128317421389157,0.149392484974272,-0.0894663968473743,-0.00580333154218123,-0.0853800163447896,-0.142296262553958,0.284079656727913,0.475222526532009,0.18240496762465,-0.0118355065195582,0.0887710630435211,0.107698126729441,-0.131366447394336,0.0257449059013715,0.0119436260848787,-0.0385716557976777,0.0389206019719781,-0.285901743642763,0.0472789046030415,0.01629659808515,-0.0911124669513923,0.027913468248429,0.206926194277294,-0.0685796787583472,0.0595634704302713,0.159568062301794,-0.227789082719245,-0.544436357755135,0.106476235273995,-0.164801455499376,-0.317917831791783,0.115524748955834,0.205569709215497,0.163983816820177,-0.103535888206648,0.496994073725422,-0.0313875754117064,-0.154774074831324,0.0973978775984893,-0.0713668709637618,-0.124287933713102,0.235133717881489,0.0327895729158132,-0.12562300841572,-0.00786082207649929,0.0977164979400871,-0.139620707329575,0.117478271717885,0.0626811072977252,0.0647414426925908,0.0655425011667786,0.0532215143940002,0.0487286436191547,0.0676498174053598,0.0870168483647173,0.245043133093596,-0.0285791008132578,0.0979090926577804],
    2014:[-0.064633407392446,-0.0233787355153476,-0.259806418746817,0.0554393305439332,-0.0561739311349206,-0.0491683230463436,-0.0168208578637522,-0.0505475989890469,0.0275190516511421,0.131940046442896,-0.00208355036982978,0.132333020735646,0.026737967914439,0.0492307692307696,0.12043274137579,0.13816054860831,0.00397891176763076,0.000994332305857126,0.0735732750049707,0.0217133833399144,-0.0640141816033096,0.122906135394984,0.0117497307353378,0.0244498777506112,0.0780487804878046,-0.422071636011617,-0.0828785122296334,0.086628618018752,-0.0616348388400525,-0.0376169174461168,0.307174201449128,0.0425742574257433,0.0226757369614502,-0.00295101318119232,-0.0757650300108232,0.0713860797144556,0.0206733608978153,-0.0815404263680138,0.00990491283676788,0.156342766673263,-0.0867108339828527,-0.381326781326781,0.294267906406457,-0.0635235732009926,0.0639296773549096,-0.155831265508684,0.0725952813067149,-0.0280280280280281,0,-0.0873318610720743,0.20759493670886,0.0218253968253967,-0.028707186695703,-0.0913332671498066,-0.106201783388438,-0.12253164556962,-0.131228214066024,0.146478287970081,0.213985870789393,0.271651964715316,0.050746559968771]

   });

dataMap.dataFinancial = dataFormatter({
    //Volatility
    2007:[2.06887946423252,2.35035969304235,2.69800142027586,2.33341299991711,2.04405619105866,1.82400430950702,1.63899998887727,1.48890990252697,1.61505313256452,2.43168326272275,2.31681011918015,2.05927034357975,2.68727324768665,3.46605664837711,2.97937155751736,2.64635478900609,2.94999580457119,2.63158642083641,2.60277763501976,2.73877499665387,2.43731958509508,2.19531759998506,1.93126913143366,1.78359206217074,4.01181307915393,3.53471806636149,3.49542373168751,3.02205014358741,2.61702664131407,9.16926001432175,9.80057381202077,8.16850400657272,6.86151976855767,6.35118534411072,5.74622182088109,8.60697167915671,7.10153044567035,6.22956535050353,5.31638203192065,4.56426211836312,3.85334902731343,3.3594259137222,2.90556007195767,2.52004195357327,2.195460516585,2.06472201246613,1.82679153586676,1.72777945784213,1.56348250442058,1.99066021840543,1.76764943874003,2.72929844763396,2.73226916440029,2.36531244936737,2.11743053988807,1.8700576671783,1.69093390254584,1.68684655117558,3.00032497248525,2.58733794054526,2.26903398586585],
    2008:[4.74049378169819,4.2335561397284,3.60133254827716,4.26572203644913,3.61484923409757,3.50562142499326,3.08865116570213,3.12135061825937,2.85507579195319,2.53474856505837,2.86198243409933,2.55610055056976,2.59926454104624,2.60148377183109,2.37986514377023,2.53797789717349,2.36231388726808,3.78493637895423,3.24387922590421,3.31160850394964,3.66723118342119,3.11846381074698,2.67269904336458,2.43491578344794,2.32004678841793,2.04642439601998,1.81308241784639,2.14966549154578,2.29502807157074,2.48228939540439,2.65324572429242,2.97453808349814,4.29823910077101,3.63997787816269,3.75948136466677,3.85628169644655,3.47822366935332,3.10225104658834,3.62539011250252,3.22023923513169,2.75146152870813,2.9471930043799,5.18176251185172,4.43869793361276,3.96869434385669,5.21959853482376,5.73532990155645,4.8963261408364,4.96976945427809,4.7789594935263,4.86949409014907,4.25715089377303,4.10647292700246,3.51692049000958,3.58314672246781,3.39070753785399,2.92105195215369,4.01446992654992,3.4342248047535,3.58174118933057,3.7252413704671],
    2009:[2.5447130326985,2.25505111212621,2.07920237534281,1.86876416107346,1.79557668480548,3.05240480920294,2.64040673998622,2.28982937449934,2.68090265944283,2.3207367474708,2.0402412562529,3.38700548595157,4.45465424891804,4.57719638571315,3.85533920344207,4.45319344399921,3.77745445110446,3.35064962705064,3.03579197886402,2.78666062152308,6.97298489038798,6.05466704214745,5.24923472248876,4.40440328196351,4.37794997624964,3.67836933178503,3.22724727686923,3.14861028445986,2.73574932562016,2.39572650255511,2.15328858874786,2.19714977125015,1.93729529814056,2.17824167435879,1.95887093897117,1.74574676297117,1.60348863274694,1.46268701714265,2.04707980909595,2.04150863921936,2.4064735121171,2.37214200369256,2.78675109414741,2.74469336490256,2.38164228261977,2.09081221117776,1.85553686325479,2.04648545789868,1.91359306494781,2.24219067946657,2.14943061543866,1.93382364157433,1.97297789542675,2.51388676713006,2.18635126151291,2.97404689264344,2.57322356411452,3.05578160566647,2.678058337683,2.40127374172366,3.28759234298925],
    2010:[3.59074068089566,3.81382424569511,3.2288788840427,3.4823219814404,3.75072199515913,3.58145139072152,3.86941081750477,3.37866678407188,4.32561122214187,5.41572923126293,10.4162164405817,8.60719511349273,7.53224114222164,6.4033356916166,5.60645707898996,4.66264918685956,3.95865048547461,3.59567332018985,4.86654430833974,4.43606964254933,4.05958360294784,3.46161548835554,2.96478202376823,3.97958348352451,3.55280346467131,3.25368966176326,2.921036728569,2.52171748290572,2.9328624956605,2.90593777745975,2.54091369073005,2.81219694880723,3.32821482023432,2.98098612912105,2.56901443190263,2.68286946679585,3.16878275524421,3.00887294467742,2.647580484168,2.4955556154803,2.35697758840452,2.17118318056837,1.96480202548779,1.81994729886344,1.66694884077798,3.55255608366178,3.34512461633596,2.99813448017486,2.61191884514771,2.31137223268042,3.65474632510345,3.10382469647242,2.69320289262849,2.41010675123534,2.55838668488031,2.23535060661595,1.98893160601244,1.80414323039033,1.91715200643597,2.36891736152007,2.15717288500774],
    2011:[5.5127961238985,4.63292671841276,3.92410542289174,4.01036753408393,3.66081116440454,3.10365156177706,2.83298348129477,2.53889233847846,2.54169111537442,2.43975388335184,2.24854928961572,1.97413861302479,2.05489733691084,1.8613878754414,1.81583688058749,1.77572617508858,1.68396895153789,1.89034313002176,2.45957317562172,2.16220772916005,1.91480532536427,2.33384761006699,2.08017065706418,1.84694829486566,1.74661259846781,1.60404809976007,1.72136755701059,2.11299438402412,2.17596541808087,1.9471993429926,1.97103702945917,1.75724224273176,1.98319290771515,4.40414817705727,4.57010260101796,3.85361190073325,3.49860645451949,4.07777167476726,3.45990382484334,2.9716234061334,5.68155434437056,4.78919403869846,4.04445339534466,4.48871892175608,4.15468011677499,3.54936254085592,3.11872771011766,2.67039493203474,2.31149218179377,2.44161343136655,2.13344275547438,2.08459498830469,1.89197201781341,1.690201290255,1.79158037784455,2.33484566331849,2.69059499816869,2.32990196530043,2.23312900723394,2.59822511529539,2.36260670922598],
    2012:[2.54224230761379,2.21140208093844,2.11349532241783,1.98957407971842,1.77359894020006,1.59673885751996,1.45436101822622,1.34093077149898,1.42598021620132,1.32298019210946,1.23523763793274,1.23948202608572,1.50870248173075,2.07322247716377,2.02034000733731,2.17252313647128,2.0000518680076,1.77583429110781,1.94862366620692,1.7355951570138,1.56736078668781,1.63208032602633,1.48492294120513,1.57686701292103,1.4858095988211,1.44486079071385,1.3689103618876,1.94678261272097,1.74295032556178,1.76332003228665,1.97346001544956,1.90145431306697,1.84735581778978,1.66214945649852,1.50479283826111,1.40228519076437,1.30078590039183,1.39209354480887,1.91224613054082,1.85885544157511,2.25083133702782,2.4112741431214,2.23759483488496,1.9735543429154,1.98895056650161,1.81412038418446,2.08044385516253,2.32761344815934,2.05030685502417,1.8785728876123,2.07734670146043,1.84344779680361,1.71407165319839,2.10691640022728,1.92408281548188,2.05908766264731,3.12111235439199,4.2452926998281,4.63405812372885,3.91425225196675,3.44607853861923],
    2013:[2.54183062012968,2.21150149351965,2.02830224951209,1.9622954059293,1.96801747041132,1.82945633797757,1.63890185695194,1.55901895747192,1.62469742934579,2.28177045714312,4.25878086294916,3.91474041250082,3.3081931221464,2.90035751405589,2.61127487625502,2.43659133601411,2.13090107060999,1.88114735852855,1.69479561313258,1.54598462308468,2.22918576864746,1.98070156312261,1.76221704158957,1.66778844961134,1.51702237678564,1.81680240020935,1.67547364355346,1.55085698493975,1.67030525301929,2.03012286447618,4.76320776803772,4.09893810121141,3.72574567831626,4.16631402036489,3.64151089250499,3.51079776747344,3.25254513576788,2.8842329100817,4.95241742124727,4.14678573590007,3.73197873111898,3.25544645050209,2.83028946311325,2.59370647515733,2.80284383297358,2.42802662729888,2.27523270427326,1.99580408865579,1.86712841062138,1.86364214764921,1.80392516137759,1.65742934122276,1.54285801699739,1.45224460818988,1.36512098249581,1.29084159308628,1.25343825241881,1.25346992092833,1.77823730750592,1.60575749603768,1.55546790108084],
    2014:[1.85341443308952,1.66319719921458,2.18055151159213,1.9501764029853,1.76669622777975,1.61253222213569,1.46785519030128,1.3748347498766,1.28244078193906,1.37503438410479,1.27507091910526,1.37017701905442,1.27829080452547,1.22186933001091,1.29753591596161,1.40391210468678,1.29828800113797,1.21364028787772,1.20004249825175,1.13974870876206,1.12777712147306,1.22828087835575,1.15900526440813,1.1081821767467,1.1224618627537,2.85441414945818,2.52721979746052,2.27182101256681,2.03044534364184,1.81350659969495,2.56936518011505,2.24861781804558,1.97903614490391,1.75831600071108,1.63905619829428,1.53720468240541,1.40903762443244,1.36871851086873,1.27095588167802,1.43619531225292,1.39914393710236,2.74841629125231,3.23966904041024,2.80708767585041,2.46154017714736,2.38706597481827,2.13735352853463,1.8927385263791,1.68919082110328,1.60262119646574,1.88805353464415,1.69020630718118,1.53540607142477,1.48674251402239,1.47718219916676,1.5068858009934,1.5527170824643,1.6317325544379,1.93828557252527,2.46357635735713,2.17161321937234]
});

dataMap.dataEstate = dataFormatter({
    //Close
    2007:[0.9031,0.9163,0.8954,0.9181,0.918,0.9166,0.9132,0.9112,0.9127,0.8983,0.8704,0.8826,0.8875,0.8614,0.8905,0.8855,0.8772,0.8547,0.8631,0.8476,0.829,0.822,0.8151,0.8152,0.8087,0.8484,0.838,0.8566,0.8505,0.8547,0.9257,0.97,0.958,0.9462,0.971,0.9495,0.8907,0.885,0.8679,0.857,0.867,0.8625,0.8538,0.8594,0.8555,0.8573,0.8474,0.8475,0.8394,0.8415,0.8615,0.8618,0.8327,0.8488,0.847,0.853,0.8521,0.8483,0.859,0.892,0.8951],
    2008:[1.6973,1.7224,1.7505,1.7395,1.8,1.8094,1.8473,1.8666,1.8259,1.8506,1.8345,1.8816,1.8996,1.8626,1.8973,1.8762,1.836,1.859,1.7819,1.7705,1.8117,1.8643,1.8701,1.8669,1.8875,1.861,1.8537,1.8519,1.8943,1.8564,1.8161,1.8564,1.8081,1.7326,1.7237,1.7684,1.8143,1.7875,1.809,1.7527,1.7316,1.7325,1.7739,1.6826,1.7009,1.6744,1.7468,1.6818,1.7012,1.7516,1.7955,1.7425,1.7663,1.7258,1.7388,1.6964,1.7281,1.7181,1.6515,1.6629,1.6202],
    2009:[1.239,1.2473,1.239,1.2514,1.2583,1.2724,1.3207,1.3271,1.325,1.2906,1.2919,1.2957,1.2442,1.1949,1.2295,1.2242,1.2665,1.2745,1.2587,1.2418,1.225,1.3078,1.3305,1.3507,1.3581,1.3935,1.3949,1.4095,1.4374,1.4467,1.4385,1.4272,1.4025,1.3995,1.3697,1.3609,1.3635,1.3558,1.3588,1.3948,1.3737,1.3401,1.3622,1.3986,1.4244,1.4197,1.4243,1.4283,1.4002,1.3861,1.354,1.3722,1.3636,1.3852,1.4234,1.4227,1.4688,1.4752,1.5175,1.5291,1.5151],
    2010:[2.7083,2.695,2.6204,2.616,2.6864,2.6109,2.6635,2.5868,2.5599,2.4625,2.3586,2.5399,2.5652,2.6209,2.5836,2.5382,2.5422,2.5236,2.4834,2.3776,2.4232,2.4676,2.4522,2.4411,2.5335,2.5688,2.6083,2.6395,2.6312,2.5596,2.5094,2.4933,2.432,2.5051,2.5351,2.5428,2.5969,2.6725,2.7187,2.7407,2.7017,2.7385,2.7097,2.69,2.667,2.683,2.5617,2.5153,2.4848,2.4694,2.4863,2.5867,2.5809,2.5962,2.5729,2.518,2.5273,2.5145,2.499,2.4558,2.5189],
    2011:[3.5644,3.736,3.7618,3.7372,3.8358,3.8997,3.8999,3.8483,3.8103,3.7405,3.7974,3.8393,3.8414,3.7748,3.7994,3.8462,3.893,3.9296,4.005,4.1163,4.1345,4.1214,4.0182,4.043,4.0317,3.9926,3.9701,3.9057,3.8132,3.746,3.725,3.7825,3.7737,3.698,3.8881,4.0029,4.0219,4.0843,4.22,4.1999,4.2224,3.9862,3.9531,3.9287,4.0577,3.9777,4.006,4.0469,4.0495,4.0478,3.9651,3.9741,4.0307,4.0024,3.9973,4.0623,3.9528,3.8522,3.8462,3.7926,3.8883],
    2012:[6.1076,6.0681,6.0779,6.157,6.2255,6.2091,6.1986,6.2073,6.217,6.3,6.3169,6.3083,6.3634,6.4811,6.6515,6.5606,6.6887,6.6263,6.6322,6.7568,6.748,6.7347,6.6387,6.6524,6.7497,6.7023,6.7627,6.8044,6.6274,6.6059,6.6979,6.8298,6.9128,6.9978,7.0191,7.021,6.987,7.0009,6.9079,6.7354,6.6518,6.8132,6.671,6.5939,6.6131,6.7145,6.668,6.5259,6.3817,6.3585,6.4091,6.281,6.2971,6.3476,6.4979,6.4461,6.3264,6.0984,6.3403,6.1336,6.1683],
    2013:[4.4779,4.5332,4.5253,4.5668,4.6254,4.6945,4.6525,4.6498,4.6101,4.5445,4.6736,4.8957,4.985,4.9791,5.0233,5.0774,5.0107,5.0236,5.0296,5.0102,5.0297,4.8859,4.909,4.917,4.8722,4.8858,4.9869,4.9527,4.9822,5.0617,4.9464,4.6771,4.7269,4.649,4.5012,4.5532,4.6468,4.723,4.6741,4.9064,4.891,4.8153,4.8622,4.8275,4.7675,4.8796,4.8956,4.8341,4.8303,4.8775,4.8094,4.8659,4.8964,4.9281,4.9604,4.9868,5.0111,5.045,5.0889,5.2136,5.1987],
    2014:[0.9902,0.9838,0.9815,0.956,0.9613,0.9559,0.9512,0.9496,0.9448,0.9474,0.9599,0.9597,0.9724,0.975,0.9798,0.9916,1.0053,1.0057,1.0058,1.0132,1.0154,1.0089,1.0213,1.0225,1.025,1.033,0.9894,0.9812,0.9897,0.9836,0.9799,1.01,1.0143,1.0166,1.0163,1.0086,1.0158,1.0179,1.0096,1.0106,1.0264,1.0175,0.9787,1.0075,1.0011,1.0075,0.9918,0.999,0.9962,0.9962,0.9875,1.008,1.0102,1.0073,0.9981,0.9875,0.9754,0.9626,0.9767,0.9976,1.0247]

    });

dataMap.dataGDP_Estate = dataMix([dataMap.dataEstate, dataMap.dataGDP]);
