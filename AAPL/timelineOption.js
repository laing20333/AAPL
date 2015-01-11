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
    //max : 60000,
    2007:[79,76,8,44,3,86,80,83,21,32,58,15,27,46,97,82,69,7,64,90,91,88,34,66,5,2,96,35,12,89,38,59,57,67,1,70,93,14,63,75,29,37,94,47,48,78,55,22,71,41,56,77,20,51,19,65,95,25,23,72,61],
    2008:[31,76,5,20,47,64,63,13,87,66,15,90,2,55,43,44,41,19,3,93,88,25,4,56,16,35,53,78,73,52,23,49,79,39,30,17,34,54,27,82,14,37,46,84,58,61,71,50,9,98,32,11,91,86,85,95,7,74,100,26,69],
    2009:[5,25,26,12,36,57,61,11,4,62,14,20,75,86,64,66,31,79,59,81,84,80,15,42,88,39,91,65,87,28,7,30,54,100,24,6,73,37,1,56,94,69,78,97,76,46,17,27,13,19,52,95,85,35,8,18,82,34,33,50,68],
    2010:[93,53,83,6,38,88,99,57,47,56,69,50,32,89,5,95,48,10,82,26,90,85,14,49,86,61,36,39,3,63,11,78,19,65,2,84,20,13,52,7,67,59,23,24,51,79,94,21,31,30,64,70,25,100,98,62,66,55,60,4,15],
    2011:[21,96,13,30,87,8,67,91,4,64,1,89,43,100,44,22,46,17,40,36,66,28,7,2,34,86,80,50,42,29,52,59,18,26,32,95,73,10,16,56,90,71,11,62,48,60,25,55,35,54,15,31,53,3,12,93,37,51,97,99,47],
    2012:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
    2013:[11,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
    2014:[41,42,43,44,45,46,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
});

dataMap.dataEstate = dataFormatter({
    //max : 3600,
    2007:[39,36,8,44,3,86,80,83,21,32,58,15,27,46,97,82,69,7,64,90,91,88,34,66,5,2,96,35,12,89,38,59,57,67,1,70,93,14,63,75,29,37,94,47,48,78,55,22,71,41,56,77,20,51,19,65,95,25,23,72,61],
    2008:[31,36,5,20,47,64,63,13,87,66,15,90,2,55,43,44,41,19,3,93,88,25,4,56,16,35,53,78,73,52,23,49,79,39,30,17,34,54,27,82,14,37,46,84,58,61,71,50,9,98,32,11,91,86,85,95,7,74,100,26,69],
    2009:[35,35,26,12,36,57,61,11,4,62,14,20,75,86,64,66,31,79,59,81,84,80,15,42,88,39,91,65,87,28,7,30,54,100,24,6,73,37,1,56,94,69,78,97,76,46,17,27,13,19,52,95,85,35,8,18,82,34,33,50,68],
    2010:[33,33,83,6,38,88,99,57,47,56,69,50,32,89,5,95,48,10,82,26,90,85,14,49,86,61,36,39,3,63,11,78,19,65,2,84,20,13,52,7,67,59,23,24,51,79,94,21,31,30,64,70,25,100,98,62,66,55,60,4,15],
    2011:[31,36,13,30,87,8,67,91,4,64,1,89,43,100,44,22,46,17,40,36,66,28,7,2,34,86,80,50,42,29,52,59,18,26,32,95,73,10,16,56,90,71,11,62,48,60,25,55,35,54,15,31,53,3,12,93,37,51,97,99,47],
    2012:[31,32,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
    2013:[31,32,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
    2014:[31,32,43,44,45,46,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
});

dataMap.dataFinancial = dataFormatter({
    //max : 3200,
    2007:[79,76,8,44,3,86,80,83,21,32,58,15,27,46,97,82,69,7,64,90,91,88,34,66,5,2,96,35,12,89,38,59,57,67,1,70,93,14,63,75,29,37,94,47,48,78,55,22,71,41,56,77,20,51,19,65,95,25,23,72,61],
    2008:[71,76,5,20,47,64,63,13,87,66,15,90,2,55,43,44,41,19,3,93,88,25,4,56,16,35,53,78,73,52,23,49,79,39,30,17,34,54,27,82,14,37,46,84,58,61,71,50,9,98,32,11,91,86,85,95,7,74,100,26,69],
    2009:[75,75,26,12,36,57,61,11,4,62,14,20,75,86,64,66,31,79,59,81,84,80,15,42,88,39,91,65,87,28,7,30,54,100,24,6,73,37,1,56,94,69,78,97,76,46,17,27,13,19,52,95,85,35,8,18,82,34,33,50,68],
    2010:[73,73,83,6,38,88,99,57,47,56,69,50,32,89,5,95,48,10,82,26,90,85,14,49,86,61,36,39,3,63,11,78,19,65,2,84,20,13,52,7,67,59,23,24,51,79,94,21,31,30,64,70,25,100,98,62,66,55,60,4,15],
    2011:[71,76,13,30,87,8,67,91,4,64,1,89,43,100,44,22,46,17,40,36,66,28,7,2,34,86,80,50,42,29,52,59,18,26,32,95,73,10,16,56,90,71,11,62,48,60,25,55,35,54,15,31,53,3,12,93,37,51,97,99,47],
    2012:[71,72,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
    2013:[71,72,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
    2014:[71,72,43,44,45,46,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
});

dataMap.dataGDP_Estate = dataMix([dataMap.dataEstate, dataMap.dataGDP]);
