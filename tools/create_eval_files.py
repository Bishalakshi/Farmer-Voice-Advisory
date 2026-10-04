import pandas as pd

# ==========================================
# 1. retrieval_dev.csv (20 questions with gold chunk IDs)
# ==========================================
dev_data = [
    # Paddy (5)
    {"question": "धान की नर्सरी में बीज उपचार कैसे करें?", "gold_ids": "KB001", "crop": "paddy"},
    {"question": "धान में तना छेदक या स्टेम बोरर के क्या लक्षण हैं?", "gold_ids": "KB003", "crop": "paddy"},
    {"question": "धान में भूरा फुदका या BPH का नियंत्रण कैसे करें?", "gold_ids": "KB005|KB006", "crop": "paddy"},
    {"question": "धान की फसल में ब्लास्ट रोग के लक्षण और दवा क्या है?", "gold_ids": "KB007", "crop": "paddy"},
    {"question": "धान की कटाई से कितने दिन पहले खेत से पानी निकाल देना चाहिए?", "gold_ids": "KB012", "crop": "paddy"},

    # Tomato (5)
    {"question": "टमाटर में सफेद मक्खी और पत्ता मरोड़ से कैसे बचें?", "gold_ids": "KB017|KB018", "crop": "tomato"},
    {"question": "टमाटर की नर्सरी में डैम्पिंग ऑफ या आद्र गलन रोग कैसे रोकें?", "gold_ids": "KB016", "crop": "tomato"},
    {"question": "टमाटर में फल छेदक इल्ली का जैविक नियंत्रण क्या है?", "gold_ids": "KB020|KB021", "crop": "tomato"},
    {"question": "टमाटर के फल नीचे से काले क्यों पड़ रहे हैं ब्लॉसम एंड रॉट?", "gold_ids": "KB022", "crop": "tomato"},
    {"question": "टमाटर के पौधों को बांस या तार से सहारा क्यों देना चाहिए?", "gold_ids": "KB025", "crop": "tomato"},

    # Onion (5)
    {"question": "प्याज की नर्सरी के लिए बीज की मात्रा और समय क्या है?", "gold_ids": "KB027", "crop": "onion"},
    {"question": "प्याज में थ्रिप्स कीट के कारण पत्तियां सफेद हो रही हैं क्या करें?", "gold_ids": "KB029|KB030", "crop": "onion"},
    {"question": "प्याज में पर्पल ब्लॉच या बैंगनी धब्बा रोग की दवा क्या है?", "gold_ids": "KB031", "crop": "onion"},
    {"question": "प्याज की खुदाई से कितने दिन पहले सिंचाई बंद करनी चाहिए?", "gold_ids": "KB035", "crop": "onion"},
    {"question": "प्याज के कंद को भंडारण में सड़ने से कैसे बचाएं?", "gold_ids": "KB036|KB037", "crop": "onion"},

    # Potato (5)
    {"question": "आलू की बुवाई के लिए बीज आलू का चयन और आकार क्या होना चाहिए?", "gold_ids": "KB038", "crop": "potato"},
    {"question": "आलू में पिछेती झुलसा या लेट ब्लाइट बीमारी के लक्षण क्या हैं?", "gold_ids": "KB041|KB042", "crop": "potato"},
    {"question": "आलू में माहू या एफिड्स से वायरस रोग का खतरा कैसे रोकें?", "gold_ids": "KB043", "crop": "potato"},
    {"question": "आलू में मिट्टी चढ़ाना या अर्दिंग अप क्यों जरूरी है?", "gold_ids": "KB044", "crop": "potato"},
    {"question": "आलू की खुदाई से पहले बेल काटना या डीहौल्मिंग क्यों करते हैं?", "gold_ids": "KB047", "crop": "potato"}
]

df_dev = pd.DataFrame(dev_data)
df_dev.to_csv("data/retrieval_dev.csv", index=False)
print(f"Created data/retrieval_dev.csv with {len(df_dev)} rows")


# ==========================================
# 2. retrieval_test.csv (20 questions with gold chunk IDs)
# ==========================================
test_data = [
    # Paddy (5)
    {"question": "धान में खैरा रोग या जिंक की कमी का क्या उपचार है?", "gold_ids": "KB010", "crop": "paddy"},
    {"question": "धान की फसल में यूरिया और खाद देने का सही समय और अनुपात क्या है?", "gold_ids": "KB009", "crop": "paddy"},
    {"question": "धान में जीवाणु झुलसा या बैक्टीरियल लीफ ब्लाइट कैसे पहचानें?", "gold_ids": "KB008", "crop": "paddy"},
    {"question": "धान की रोपाई के बाद खरपतवार नियंत्रण के लिए कौन सा स्प्रे करें?", "gold_ids": "KB014", "crop": "paddy"},
    {"question": "श्री विधि (SRI) से धान की रोपाई में कितने दिन की पौध लगाते हैं?", "gold_ids": "KB002", "crop": "paddy"},

    # Tomato (5)
    {"question": "टमाटर में अगेती झुलसा या अर्ली ब्लाइट रोग के लक्षण क्या हैं?", "gold_ids": "KB019", "crop": "tomato"},
    {"question": "हाइब्रिड टमाटर में खाद और एनपीके की कितनी मात्रा देनी चाहिए?", "gold_ids": "KB023", "crop": "tomato"},
    {"question": "टमाटर में ड्रिप सिंचाई से क्या फायदा होता है?", "gold_ids": "KB024|KB066", "crop": "tomato"},
    {"question": "दूर मंडी में भेजने के लिए टमाटर किस अवस्था में तोड़ना चाहिए?", "gold_ids": "KB026", "crop": "tomato"},
    {"question": "टमाटर में सफेद मक्खी के लिए पीला चिपचिपा ट्रैप कैसे लगाएं?", "gold_ids": "KB017|KB063", "crop": "tomato"},

    # Onion (5)
    {"question": "प्याज की पौध खेत में कितने दिन बाद रोपनी चाहिए?", "gold_ids": "KB028", "crop": "onion"},
    {"question": "प्याज में गंध और कंद की चमक बढ़ाने के लिए कौन सी खाद जरूरी है?", "gold_ids": "KB033", "crop": "onion"},
    {"question": "प्याज में जड़ गलन या बेसल रॉट रोग का क्या कारण है?", "gold_ids": "KB032", "crop": "onion"},
    {"question": "प्याज की खुदाई कब करनी चाहिए गर्दन झुकने का क्या मतलब है?", "gold_ids": "KB036", "crop": "onion"},
    {"question": "प्याज में सिंचाई किस अवस्था में सबसे अधिक जरूरी होती है?", "gold_ids": "KB034", "crop": "onion"},

    # Potato (5)
    {"question": "आलू के बीज कंद को बुवाई से पहले अंकुरित कैसे करें?", "gold_ids": "KB039", "crop": "potato"},
    {"question": "आलू की बुवाई के लिए कतार से कतार की दूरी क्या होनी चाहिए?", "gold_ids": "KB040", "crop": "potato"},
    {"question": "कोल्ड स्टोरेज में बीज आलू किस तापमान पर रखा जाता है?", "gold_ids": "KB049", "crop": "potato"},
    {"question": "आलू की फसल में पानी देने का सही तरीका क्या है मेंड़ के ऊपर पानी जाना चाहिए?", "gold_ids": "KB046", "crop": "potato"},
    {"question": "आलू को हरा होने से बचाने के लिए क्या सावधानी बरतें?", "gold_ids": "KB044", "crop": "potato"}
]

df_test = pd.DataFrame(test_data)
df_test.to_csv("data/retrieval_test.csv", index=False)
print(f"Created data/retrieval_test.csv with {len(df_test)} rows")


# ==========================================
# 3. threshold_val.csv (20 answerable + 20 out-of-scope)
# ==========================================
threshold_data = [
    # 20 Answerable questions
    {"question": "धान में तना छेदक के लिए प्रकाश प्रपंच या लाइट ट्रैप कैसे लगाएं?", "gold_ids": "KB003", "crop": "paddy"},
    {"question": "धान में यूरिया तीन बार में कब-कब डालना चाहिए?", "gold_ids": "KB009", "crop": "paddy"},
    {"question": "धान में बीपीएच की रोकथाम के लिए कौन सी दवा स्प्रे करें?", "gold_ids": "KB006", "crop": "paddy"},
    {"question": "धान की पकी फसल को कितने नमी प्रतिशत पर सुरक्षित रखना चाहिए?", "gold_ids": "KB013|KB071", "crop": "paddy"},
    {"question": "टमाटर में सफेद मक्खी से पत्ती मुड़ने का वायरस कैसे फैलता है?", "gold_ids": "KB017", "crop": "tomato"},
    {"question": "टमाटर में फल छेदक के लिए फेरोमोन ट्रैप कितनी संख्या में लगाएं?", "gold_ids": "KB020|KB064", "crop": "tomato"},
    {"question": "टमाटर में नीम तेल का छिड़काव कब और कैसे करें?", "gold_ids": "KB018|KB062", "crop": "tomato"},
    {"question": "टमाटर की नर्सरी में फफूंद से पौधे गिरने का क्या उपाय है?", "gold_ids": "KB016", "crop": "tomato"},
    {"question": "प्याज की फसल में सल्फर डालने से क्या लाभ होता है?", "gold_ids": "KB033", "crop": "onion"},
    {"question": "प्याज में थ्रिप्स के नियंत्रण के लिए नीला चिपचिपा जाल कैसे काम करता है?", "gold_ids": "KB030|KB063", "crop": "onion"},
    {"question": "प्याज की खुदाई के बाद धूप में सुखाने का क्या तरीका है?", "gold_ids": "KB036", "crop": "onion"},
    {"question": "प्याज के बीज का शोधन किस कवकनाशी से करना चाहिए?", "gold_ids": "KB027", "crop": "onion"},
    {"question": "आलू में लेट ब्लाइट रोकने के लिए मौसम खराब होने से पहले क्या छिड़कें?", "gold_ids": "KB042", "crop": "potato"},
    {"question": "आलू के बीज को कोल्ड स्टोर से निकालने के बाद क्या करें?", "gold_ids": "KB039", "crop": "potato"},
    {"question": "आलू में पहली बार मिट्टी कब चढ़ानी चाहिए?", "gold_ids": "KB044", "crop": "potato"},
    {"question": "आलू में कंद बनते समय सिंचाई का क्या महत्व है?", "gold_ids": "KB046", "crop": "potato"},
    {"question": "ओडिशा की अम्लीय या लाल मिट्टी में चूना कितना डालना चाहिए?", "gold_ids": "KB055", "crop": "general"},
    {"question": "खेत में कीटनाशक छिड़कते समय मौसम की क्या सावधानी रखनी चाहिए?", "gold_ids": "KB058", "crop": "general"},
    {"question": "ट्राइकोडर्मा जैविक फफूंदनाशी का प्रयोग बीज और मिट्टी में कैसे करें?", "gold_ids": "KB065", "crop": "general"},
    {"question": "नीम के बीज की गिरी से 5 प्रतिशत का घोल कैसे बनाएं?", "gold_ids": "KB062", "crop": "general"},

    # 20 Out-of-scope questions (gold_ids empty)
    {"question": "भारत और ऑस्ट्रेलिया के बीच क्रिकेट मैच कौन जीतेगा?", "gold_ids": "", "crop": ""},
    {"question": "मनुष्य के सिरदर्द और बुखार के लिए कौन सी पेरासिटामोल गोली लें?", "gold_ids": "", "crop": ""},
    {"question": "मोटरसाइकिल का पंचर टायर खुद घर पर कैसे ठीक करें?", "gold_ids": "", "crop": ""},
    {"question": "आज मुंबई में 24 कैरेट सोने का क्या भाव रहा?", "gold_ids": "", "crop": ""},
    {"question": "ऑनलाइन पासपोर्ट बनवाने के लिए सरकारी वेबसाइट पर क्या दस्तावेज चाहिए?", "gold_ids": "", "crop": ""},
    {"question": "फ्रांस देश की राजधानी का क्या नाम है?", "gold_ids": "", "crop": ""},
    {"question": "घर पर बिना अंडे का चॉकलेट केक कैसे बेक करें?", "gold_ids": "", "crop": ""},
    {"question": "ब्रिटेन के वर्तमान प्रधानमंत्री कौन हैं?", "gold_ids": "", "crop": ""},
    {"question": "शतरंज के खेल में वजीर और घोड़े की क्या चाल होती है?", "gold_ids": "", "crop": ""},
    {"question": "सॉफ्टवेयर इंजीनियर की नौकरी के लिए बायोडाटा या रिज्यूम कैसे बनाएं?", "gold_ids": "", "crop": ""},
    {"question": "भुवनेश्वर से नई दिल्ली जाने वाली राजधानी एक्सप्रेस ट्रेन का समय क्या है?", "gold_ids": "", "crop": ""},
    {"question": "एयरटेल मोबाइल फोन नंबर पर 299 रुपये का ऑनलाइन रिचार्ज कैसे करें?", "gold_ids": "", "crop": ""},
    {"question": "पिछला फीफा फुटबॉल विश्व कप किस देश ने जीता था?", "gold_ids": "", "crop": ""},
    {"question": "चक्रवृद्धि ब्याज निकालने का गणितीय सूत्र क्या होता है?", "gold_ids": "", "crop": ""},
    {"question": "शेयर बाजार में ट्रेडिंग करने के लिए डीमैट खाता कैसे खोलें?", "gold_ids": "", "crop": ""},
    {"question": "बाल झड़ने से रोकने के लिए सबसे अच्छा शैम्पू कौन सा है?", "gold_ids": "", "crop": ""},
    {"question": "पृथ्वी से मंगल ग्रह की कुल दूरी कितनी है?", "gold_ids": "", "crop": ""},
    {"question": "इलेक्ट्रिक कार पेट्रोल कार से कैसे अलग काम करती है?", "gold_ids": "", "crop": ""},
    {"question": "मानव शरीर में उच्च रक्तचाप या हाई बीपी के क्या लक्षण हैं?", "gold_ids": "", "crop": ""},
    {"question": "गोवा में छुट्टियां बिताने के लिए सस्ता होटल कैसे बुक करें?", "gold_ids": "", "crop": ""},
]

df_val = pd.DataFrame(threshold_data)
df_val.to_csv("data/threshold_val.csv", index=False)
print(f"Created data/threshold_val.csv with {len(df_val)} rows ({sum(df_val.gold_ids != '')} answerable + {sum(df_val.gold_ids == '')} out-of-scope)")


# ==========================================
# 4. e2e_test.csv (30 questions: 8 abstain, 6 weather, 4 price, 12 normal)
# ==========================================
e2e_data = [
    # 8 Abstain (Out of scope / unverified queries)
    {"question": "क्या आज रात का आईपीएल क्रिकेट मैच कोलकाता जीतेगा?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "मेरे पेट में बहुत दर्द है कौन सी एंटीबायोटिक दवा खानी चाहिए?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "स्मार्टफोन का स्क्रीन टूट गया है इसे कैसे बदलें?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "क्या डीडीटी को घर के अंदर मच्छर मारने के लिए सीधे छिड़क सकते हैं?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "शेयर बाजार में कल रिलायंस का शेयर ऊपर जाएगा या नीचे?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "हवाई जहाज का टिकट आईआरसीटीसी से कैसे कैंसिल करें?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "क्या आलू के खेत में डीजल डालने से कीड़े मर जाते हैं?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},
    {"question": "गूगल और माइक्रोसॉफ्ट में कौन सी कंपनी बड़ी है?", "gold_ids": "", "crop": "", "expected_behaviour": "abstain", "place": ""},

    # 6 Weather (with place: Bhubaneswar)
    {"question": "Bhubaneswar में आज बारिश होने की क्या संभावना है?", "gold_ids": "", "crop": "general", "expected_behaviour": "answer", "place": "Bhubaneswar"},
    {"question": "क्या आज Bhubaneswar में तेज धूप रहेगी या बादल छाएंगे?", "gold_ids": "", "crop": "general", "expected_behaviour": "answer", "place": "Bhubaneswar"},
    {"question": "Bhubaneswar में कल का तापमान कितना रहने का अनुमान है?", "gold_ids": "", "crop": "general", "expected_behaviour": "answer", "place": "Bhubaneswar"},
    {"question": "Bhubaneswar में क्या अगले 24 घंटे में भारी बारिश की चेतावनी है?", "gold_ids": "", "crop": "general", "expected_behaviour": "answer", "place": "Bhubaneswar"},
    {"question": "क्या Bhubaneswar में आज हवा की गति तेज रहेगी क्या कीटनाशक छिड़कें?", "gold_ids": "KB058", "crop": "general", "expected_behaviour": "answer", "place": "Bhubaneswar"},
    {"question": "Bhubaneswar में क्या इस हफ्ते चक्रवात या आंधी का कोई अलर्ट है?", "gold_ids": "KB059", "crop": "general", "expected_behaviour": "answer", "place": "Bhubaneswar"},

    # 4 Price questions
    {"question": "मंडी में आज धान का सरकारी न्यूनतम समर्थन मूल्य क्या चल रहा है?", "gold_ids": "KB069", "crop": "paddy", "expected_behaviour": "answer", "place": ""},
    {"question": "मंडी में टमाटर का थोक भाव आज क्या है?", "gold_ids": "", "crop": "tomato", "expected_behaviour": "answer", "place": ""},
    {"question": "इस सप्ताह प्याज का मंडी भाव कैसा चल रहा है?", "gold_ids": "", "crop": "onion", "expected_behaviour": "answer", "place": ""},
    {"question": "मंडी में आलू का रेट आज क्या मिल रहा है?", "gold_ids": "", "crop": "potato", "expected_behaviour": "answer", "place": ""},

    # 12 Normal crop advisory questions (Answerable from KB)
    {"question": "धान में तना छेदक के कारण सफेद बाली आ रही है क्या करें?", "gold_ids": "KB003|KB004", "crop": "paddy", "expected_behaviour": "answer", "place": ""},
    {"question": "धान की फसल में बीपीएच की रोकथाम कैसे करें?", "gold_ids": "KB005|KB006", "crop": "paddy", "expected_behaviour": "answer", "place": ""},
    {"question": "धान में खैरा रोग के लिए जिंक सल्फेट कितना स्प्रे करें?", "gold_ids": "KB010", "crop": "paddy", "expected_behaviour": "answer", "place": ""},
    {"question": "टमाटर के पत्ते मुड़ रहे हैं सफेद मक्खी का क्या इलाज है?", "gold_ids": "KB017|KB018", "crop": "tomato", "expected_behaviour": "answer", "place": ""},
    {"question": "टमाटर में फल छेदक इल्ली से बचाव के उपाय बताएं?", "gold_ids": "KB020|KB021", "crop": "tomato", "expected_behaviour": "answer", "place": ""},
    {"question": "टमाटर में ब्लॉसम एंड रॉट के लिए कौन सी खाद या कैल्शियम स्प्रे करें?", "gold_ids": "KB022", "crop": "tomato", "expected_behaviour": "answer", "place": ""},
    {"question": "प्याज की फसल में थ्रिप्स कीट से पत्तियां सूख रही हैं क्या उपाय है?", "gold_ids": "KB029|KB030", "crop": "onion", "expected_behaviour": "answer", "place": ""},
    {"question": "प्याज की खुदाई से पहले पानी क्यों बंद कर देना चाहिए?", "gold_ids": "KB035", "crop": "onion", "expected_behaviour": "answer", "place": ""},
    {"question": "प्याज में बैंगनी धब्बा रोग पर्पल ब्लॉच कैसे रोकें?", "gold_ids": "KB031", "crop": "onion", "expected_behaviour": "answer", "place": ""},
    {"question": "आलू में लेट ब्लाइट रोग के क्या लक्षण हैं और कौन सी दवा छिड़कें?", "gold_ids": "KB041|KB042", "crop": "potato", "expected_behaviour": "answer", "place": ""},
    {"question": "आलू को धूप से हरा होने से बचाने के लिए मिट्टी कब चढ़ाएं?", "gold_ids": "KB044", "crop": "potato", "expected_behaviour": "answer", "place": ""},
    {"question": "आलू की खुदाई से पहले बेल काटना क्यों जरूरी है?", "gold_ids": "KB047", "crop": "potato", "expected_behaviour": "answer", "place": ""}
]

df_e2e = pd.DataFrame(e2e_data)
df_e2e.to_csv("data/e2e_test.csv", index=False)
print(f"Created data/e2e_test.csv with {len(df_e2e)} rows:")
print("  Abstain:", sum(df_e2e.expected_behaviour == "abstain"))
print("  Weather:", sum(df_e2e.place != ""))
print("  Price  :", sum(df_e2e.question.str.contains("भाव|मूल्य|रेट")))
print("  Normal :", sum((df_e2e.expected_behaviour == "answer") & (df_e2e.place == "") & (~df_e2e.question.str.contains("भाव|मूल्य|रेट"))))
