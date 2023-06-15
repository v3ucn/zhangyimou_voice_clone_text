from pathlib import Path
import soundfile as sf
import os
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output

# 音色模型的路径
am_inference_dir = "./master"

# 声码器的路径
voc_inference_dir_pwgan = "./pwgan" 

# 声码器的路径
voc_inference_dir_wavernn = "./wavernn" 



# 克隆音频生成的路径
wav_output_dir = "./output"

# 选择设备[gpu / cpu]，默认选择gpu， 
device = "gpu"

# 想要生成的文本和对应文件名

text_dict = {
    "1": "我原来想拿中石油的offer",
    "2": "是不是很大胆",
    "3": "中石油",
    "4": "国企天花板",
    "5": "就是中石油",
    "6": "出差可以逛太古里",
    "7": "太爽了",
    "8": "我最早准备面试的时候",
    "9": "跟所有同学说的只面中石油",
    "10": "所有的同学，包括亲戚，朋友，他们所有人很兴奋",
    "11": "我女朋友也很兴奋",
    "12": "中石油",
    "13": "一直说的是去中石油",
    "14": "我一直在做去中石油的准备",
    "15": "当时我面试的时候",
    "16": "我说试用期只要20天",
    "17": "或者只要25天",
    "18": "两周到三周",
    "19": "hr说为什么?",
    "20": "我说很简单",
    "21": "我每天飞四川",
    "22": "单程两个小时",
    "23": "早上去一次",
    "24": "晚上去一次",
    "25": "每天去两次",
    "26": "我坚持10天",
    "27": "20次",
    "28": "就是20次",
    "29": "成都太古里",
    "30": "哇简直太爽了",
    "31": "逛街",
    "32": "去10天就够了",
    "33": "然后前面的十天在北京",
    "34": "上班",
    "35": "严格地上班",
    "36": "我说试用期只要二十天",
    "37": "咱试用期就结束了",
    "38": "哇hr说真的太厉害",
    "39": "就挑战性太大了",
    "40": "一天都不能请假啊",
    "41": "但是后来我还是放弃了，哈哈哈",


    "42": "你知道为什么",
    "43": "我研究了大量的员工去成都的案例",
    "44": "嗯，也有一些基层员工",
    "45": "还有尤其是最近一段时间一些比较大胆的行为",
    "46": "就是牵手那个我也看了",
    "47": "我专门看",
    "48": "研究",
    "49": "就一直，我就一直下不了决心",
    "50": "其实我真的非常想去啊，内心深处非常想",
    "51": "你知道最大问题是什么，当然这是一个专业问题，简单地说最大问题就是街拍",
    "52": "就是街拍",
    "53": "因为你去了他就拍你啊",
    "54": "就没有办法",
    "55": "对一个员工",
    "56": "对一个向往太古里的员工",
    "57": "一个经常逛太古里的员工来说",
    "58": "他给你来一个街拍",
    "59": "全给你拍下来",
    "60": "上传抖音",
    "61": "因为你不能蹭蹭蹭蹭",
    "62": "逛的太快啊",
    "63": "不能啊",
    "64": "你从南边到北边",
    "65": "你中间得逛啊",
    "66": "就拍了",
    "67": "就拍了",
    "68": "第一是街拍避免不了",
    "69": "无论怎么样",
    "70": "我想来想去",
    "71": "因为我算个内行嘛",
    "72": "我不去了，我就知道街拍跑不了",
    "73": "街拍，避免不了",

    "74": "第二个",
    "75": "你的工资会全都损失了",
    "76": "不是损失一半的工资，一半无所谓",
    "77": "是全部的工资，奖金，绩效，年终奖全都没有了",
    "78": "然后你还得停职",
    "79": "就很尴尬啊",
    "80": "这样子就不好混了",
    "81": "真的不好混了",
    "82": "最后我差不多一个多月的思想斗争",
    "83": "那是个重大决定",
    "84": "因为我都是按照去中石油准备的",
    "85": "背面试题呢",
    "86": "后来说放弃",
    "87": "我自己决定放弃",
    "88": "一个人做的决定，一个人的思考",
    "89": "一个多月以后我放弃了，我第一个电话打给人力，我说我放弃去中石油。他，啊这，就不能接受",
    "90": "他已经完全沉浸到去太古里当中去了，你知道吧",
    "91": "就想着太好了，就喜欢的不得了",
    "92": "怎么可能就过来说服我",
    "93": "我说你不用跟我说",
    "94": "你都不太清楚",
    "95": "反正去中石油",
    "96": "说怎么可能，你能做到，就开始给我忽悠",
    "97": "我放弃了",
    "98": "然后我跟女朋友说放弃",
    "99": "哎呀，她说她把包包裙子都买了，这那的",
    "100": "所有人，大家都觉得太遗憾了。",
    "101": "然后跟老板说",
    "102": "最有意思是跟老板说",
    "103": "说真的不去中石油了",
    "104": "哎呀，哎呀",
    "105": "就觉着好像就没劲了，哈哈哈",
    "106": "说你不是开玩笑吧",
    "107": "哎呀就觉得，好像不想要我了似的",
    "108": "开玩笑啊，开玩笑",
    "109": "就所有人都沮丧而失落",
    "110": "就我看到大家的反应",
    "111": "我也很难过，很难过",
    "112": "我我，我后来还是放弃了",
    "113": "放弃了，嗯",
    "114": "所以中石油offer是一个学习",
    "115": "它对于一个追求太古里的一个员工来说",
    "116": "它是破坏性的",
    "117": "你去了中石油又能怎么样呢?",
    "118": "你丢掉了信仰",
    "119": "丢掉了人格啊",
    "120": "孰重孰轻啊",
    "121": "所以我在学习",
    "122": "我在学习做一个合格员工的思考",
    "123": "这就是我的，遗憾",
    "124": "但也许是我的一个清醒",
    "125": "或者学习的心得",
}

# frontend
frontend = get_frontend(
    lang="mix",
    phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
    tones_dict=None
)

# am_predictor
am_predictor = get_predictor(
    model_dir=am_inference_dir,
    model_file="fastspeech2_mix" + ".pdmodel",
    params_file="fastspeech2_mix" + ".pdiparams",
    device=device)

# voc_predictor
voc_predictor_pwgan = get_predictor(
    model_dir=voc_inference_dir_pwgan,
    model_file="pwgan_aishell3" + ".pdmodel",    
    params_file="pwgan_aishell3" + ".pdiparams",
    device=device)


voc_predictor_wavernn = get_predictor(
    model_dir=voc_inference_dir_wavernn,
    model_file="wavernn_csmsc" + ".pdmodel",    
    params_file="wavernn_csmsc" + ".pdiparams",
    device=device)

output_dir = Path(wav_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

sentences = list(text_dict.items())


def clone(voc_predictor):

    merge_sentences = True
    fs = 24000
    for utt_id, sentence in sentences:
        am_output_data = get_am_output(
            input=sentence,
            am_predictor=am_predictor,
            am="fastspeech2_mix",
            frontend=frontend,
            lang="mix",
            merge_sentences=merge_sentences,
            speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
            spk_id=0, )
        wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)
        # 保存文件
        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)


if __name__ == '__main__':
    
    clone(voc_predictor_pwgan)
