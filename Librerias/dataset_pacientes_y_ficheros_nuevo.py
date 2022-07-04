def get_dataset():
    # Pacientes y sus ficheros correspondientes

    # AÑO 2018
    # Paciente 559, 2018
    P1 = ["167", "168", "169", "170", "171", "175", "182", "183", "184", "187", "188", "189", "190",
          "193", "194", "195", "196", "200", "202", "203", "207"]  # Paciente 559 training
    P2 = ["208", "209", "211", "214"]  # Paciente 559 testing
    # # Paciente 563, 2018
    # P3 = ["219","220","221","222","224","225","226","227","228","229","231","232","235","239","240","241","246","248","251","253","254","255","258","260","261","262","263"] #Paciente 563 training sin hr
    P3 = ["225", "226", "227", "228", "229", "231", "232", "235", "239", "241", "246", "248", "251",
          "253", "254", "255", "258", "261", "262", "263"]  # Paciente 563 training con hr
    P4 = ["265", "266", "269", "270", "271"]  # Paciente 563 testing con hr
    # P4 = ["265", "266", "268", "269", "270", "271"]  # Paciente 563 testing sin hr
    # # Paciente 570, 2018
    P5 = ["332", "333", "334", "335", "336", "338", "339", "340", "344", "345", "346",
          "349", "351", "352", "354", "355", "356", "357", "358", "360", "361", "365", "366",
          "367", "371"]  # Paciente 570 training
    P6 = ["373", "375", "377", "381"]  # Paciente 570 testing con hr
    # P6 = ["372", "373", "374", "375", "377", "379", "380", "381"]  # Paciente 570 testing sin hr
    # # Paciente 575, 2018
    # P7 = ["383","384","385","388","389","390","392","393","394","395","396","397","399","400","401","402","403","405","406","407","408","409","411","412","413","414","416","417","418","419","420","422","423","424","425","426","427"] #Paciente 575 training sin hr
    P7 = ["383", "384", "385", "388", "389", "390", "392", "393", "395", "396", "397", "399", "400", "401",
          "402", "403", "405", "406", "407", "408", "409", "411", "412", "413", "414", "416", "418", "420", "423",
          "424", "425", "426"]  # Paciente 575 training con hr
    P8 = ["428", "429", "430", "431", "432", "435", "436"]  # Paciente 575 testing
    # # Paciente 588, 2018
    P9 = ["496", "497", "498", "499", "500", "502", "503", "504", "505", "506", "508", "509", "510", "511",
          "512", "514", "515", "516", "519", "520", "521", "522", "523", "525", "526", "527", "528",
          "529", "531", "532", "533", "534", "535", "537", "538", "539", "540"]  # Paciente 588 training
    P10 = ["541", "543", "544", "545", "546", "547"]  # Paciente 588 testing con hr
    # P10 = ["541","542","543","544","545","546","547","548","550"] #Paciente 588 testing sin hr
    # Paciente 591, 2018
    # P11 = ["552","553","554","555","556","557","558","560","561","562","563","564","565","566","567","568","569","570","571","572","573","574","575","581","582","587","588","589","590","592","593","594","595"] #Paciente 591 training sin hr
    P11 = ["552", "553", "554", "555", "556", "558", "560", "561", "563", "564", "566", "567", "568",
           "569", "572", "573", "574", "575", "581", "587", "588", "589", "590", "592", "594",
           "595"]  # Paciente 591 training con hr
    P12 = ["596", "598", "599", "600", "601", "603"]  # Paciente 591 testing con hr
    # P12 = ["596", "597", "598", "599", "600", "601", "602", "603", "604"]  # Paciente 591 testing sin hr

    # # AÑO 2020
    # # Paciente 540, 2020
    P13 = ["002", "003", "004", "005", "006", "008", "009", "010", "011", "012", "015", "016", "017", "018", "019",
           "021", "022", "023", "024", "025", "027", "028", "029", "030", "031", "034", "035", "036", "038",
           "041", "042", "044"]  # Paciente 540 training
    P14 = ["047", "048", "049", "050", "051", "053", "054", "055", "056"]  # Paciente 540 testing
    # # Paciente 544, 2020
    P15 = ["059", "059", "060", "061", "062", "065", "066", "067", "068", "069",
           "073", "074", "075", "081", "086", "087", "089", "090", "091", "092", "095",
           "096", "097", "099", "101"]  # Paciente 544 training![](dataset/2020/fig/glucosa/Paciente 544-fichero 88.png)
    P16 = ["102", "104", "107", "108", "109", "110", "111"]  # Paciente 544 testing
    # # Paciente 552, 2020
    P17 = ["114", "115", "117", "119", "120", "123", "124", "129", "130",
           "132", "133", "137", "139", "140", "141",
           "144", "145", "146", "147", "148"]  # Paciente 552 training
    P18 = ["154", "155", "156", "162"]  # Paciente 552 testing
    # # Paciente 567, 2020
    P19 = ["277", "281", "282", "284", "285", "288", "289", "290",
           "296", "301", "303", "308", "316", "319"]  # Paciente 567 training
    P20 = ["322", "323", "328", "330"]  # Paciente 567 testing
    # # Paciente 584, 2020
    P21 = ["439", "440", "441", "442", "443", "445", "447", "448", "449", "450", "451", "452", "454", "455",
           "456", "457", "458", "460", "461", "462", "463", "464", "466", "467", "468", "469", "471", "473",
           "474", "475", "476", "477", "478", "480", "483"]  # Paciente 584 training
    P22 = ["484", "485", "486", "487", "488", "491", "492", "493"]  # Paciente 584 testing
    # # Paciente 596, 2020
    P23 = ["607", "611", "612", "613", "614", "619", "620",
           "621", "623", "625", "626", "627", "628", "631", "632", "633", "634", "635",
           "638", "639", "640", "641", "642", "645", "646", "647", "648",
           "653", "655"]  # Paciente 596 training
    P24 = ["657", "659", "660", "661", "663", "664"]  # Paciente 596 testing

    '''train_data = P1 + P3 + P5 + P7 + P9 + P11 + P13 + P15 + P17 + P19 + P21 + P23 + P2 + P4 + P6 + P8 + P10 + P12  # dataset of training (impares)
    #   train_data = P1  # dataset of training
    eval_data = P2 + P4 + P6 + P8 + P10 + P12  # dataset of evaluation (pares, excepto P2 que dejo para test)
    #   eval_data = P2  # dataset of evaluation
    test_data = P14 + P16 + P18 + P20 + P22 + P24 # dataset of testing (to prevent overfitting)
    all_dataset = train_data + eval_data + test_data'''

    train_data = P1 + P3 + P5 + P7 + P9 + P11  # dataset of training (impares)
    #   train_data = P1  # dataset of training
    eval_data = P1 # dataset of evaluation (pares, excepto P2 que dejo para test)
    #   eval_data = P2  # dataset of evaluation
    test_data = P2 + P4 + P6 + P8 + P10 + P12  # dataset of testing (to prevent overfitting)
    all_dataset = train_data + eval_data + test_data

    '''train_data = P1  # dataset of training (impares)
    #   train_data = P1  # dataset of training
    eval_data = P1  # dataset of evaluation (pares, excepto P2 que dejo para test)
    #   eval_data = P2  # dataset of evaluation
    test_data = P2  # dataset of testing (to prevent overfitting)
    all_dataset = train_data + eval_data + test_data'''

    return all_dataset, train_data, eval_data, test_data


def main():
    #    all_dataset,train_data, eval_data,test_data=dataset_pacientes_ficheros()
    all_dataset, train_data, eval_data, test_data = get_dataset()
    print('El conjunto de entrenamiento es:')
    print(train_data)
    print('El conjunto de evaluacion es:')
    print(eval_data)
    print('El conjunto de test es:')
    print(test_data)


#    print(all_dataset)

if __name__ == '__main__':
    main()