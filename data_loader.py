import json
import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
import numpy as np
# import prettytable as pt
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open("./data/graph.json", "r", encoding="utf-8") as f:
    graph = json.loads(f.read())

with open("./data/attr.json", "r", encoding="utf-8") as f:
    attr = json.loads(f.read())

with open("./data/type2nature.json", "r", encoding="utf-8") as f:
    t2n = json.loads(f.read())

class MyDataset(Dataset):
    def __init__(self, src_list, src_mask_list, tgt_list, tgt_mask_list, video_atts_list, video_hidden_list, video_query_list, vg_list, relation_list, link_list, test_list, video_id_list, docs):
        self.src_list = src_list
        self.src_mask_list = src_mask_list
        self.tgt_list = tgt_list
        self.tgt_mask_list = tgt_mask_list
        self.video_atts_list = video_atts_list
        self.video_hidden_list = video_hidden_list
        self.video_query_list = video_query_list
        self.vg_list = vg_list
        self.relation = relation_list
        self.links = link_list
        self.test_list = test_list
        self.video_id_list = video_id_list
        self.docs = docs

    def __getitem__(self, item):
        return torch.LongTensor(self.src_list[item]), \
               torch.LongTensor(self.src_mask_list[item]), \
               torch.LongTensor(self.tgt_list[item]), \
               torch.LongTensor(self.tgt_mask_list[item]), \
               torch.LongTensor(self.video_atts_list[item]), \
               torch.FloatTensor(self.video_hidden_list[item]), \
               torch.FloatTensor(self.video_query_list[item]), \
               self.vg_list[item], \
               self.relation[item], \
               self.links[item], \
               self.test_list[item], \
               self.video_id_list[item], \
               self.docs[item]

    
    def __len__(self):
        return len(self.src_list)

def dataPro_m3d_code(rel, chain, v_id, lang):
    output = ""
    # chain_dic = {}
    # rel_list = []
    for chain_id, v in chain.items():
        # s = ", ".join(["\"" + item["text"] + "\"" for item in v["link"]])
        output += f'chain_dic["{chain_id}"] = {json.dumps([list(set([item["text"] for item in v["link"]])), t2n[lang]["entity_type_dic"][v["type"]]], ensure_ascii=False)}'
        output += "\n"
        # chain_dic[chain_id] = [item["text"] for item in v["link"]]
    r_dic = {}
    for r in rel:
        t = r["type"]
        if t not in r_dic:
            r_dic[t] = [[r["link1"], r["link2"]]]
        else:
            r_dic[t].append([r["link1"], r["link2"]])
        # ss = ", ".join(["\"" + r["link1"] + "\"", "\"" + r["link2"] + "\"", "\"" + relation_type_dic[r["type"]] + "\""])
    for k, v in r_dic.items():
        output += f'relation_dic["{t2n[lang]["relation_type_dic"][k]}"] = {json.dumps(v, ensure_ascii=False)}'
        output += "\n"
        # rel_list.append([r["link1"], r["link2"], relation_type_dic[r["type"]]])
    img_list = []
    for img in os.listdir(f"./img/{v_id}"):
        if "txt" not in img:
            img_list.append(int(img.split(".")[0]))
    img_list = sorted(img_list)
    for i, img in enumerate(img_list):
        with open(f"./img/{v_id}/{img}.txt", "r", encoding="utf-8") as f:
            new = []
            for line in f.read().split("\n"):
                new_line = []
                for item in line.split(" "):
                    if item in t2n[lang]["entity_type_dic"]:
                        new_line.append(t2n[lang]["entity_type_dic"][item])
                    else:
                        new_line.append(str(round(float(item), 1)))
                new.append(new_line)
            output += f'grounding_dic["Img{i}"] = {json.dumps(new, ensure_ascii=False)}'
            output += "\n"
    return output

def process_bert(data, tokenizer, myconfig, name="train"):
    src_list = []
    src_mask_list = []
    tgt_list = []
    tgt_mask_list = []
    link_list = []
    relation_list = []
    test_list = []
    video_id_list = []
    docs = []
    data_max_len = 0
    video_atts_list = []
    video_hidden_list = []
    video_query_list = []
    vg_list = []
    video_path = "./feature/"
    for item in tqdm(data):
        prompt_1, prompt_2 = "", ""
        text = item["doc"]
        video_id = item["video_id"]
        relation = item["relation"]
        entityLink = item["entityLink"]
        if myconfig.dataset == "en":
            prompt_1 = """def information_extraction(input_text, scene_graph, input_image, entity_attribute):
                \"\"\"
                    first , extract entities from text .
                    second , extract entity chains base on entities .
                    third , extract entity chains relation based on entity chains .
                    fourth , inferring the visual area coordinate and type in the image .
                \"\"\"
                input_text = \"""" + text + """\"
                entity_attribute = """ + json.dumps(attr["en"][video_id] if video_id in attr["en"] else {}, ensure_ascii=False) + """
                scene_graph = """ + json.dumps({f"Img{i}": item[-1] for i, item in enumerate(sorted(graph["en"][video_id].items(), key=lambda x: int(x[0].split('.')[0])))} if video_id in graph["en"] else {}, ensure_ascii=False) + """
                input_image = """

            prompt_2 = """\nentity_dic = {}
                chain_dic = {}
                relation_dic = {}
                grounding_dic = {}
                # extacted entities , entity chains , relations and visual area"""

            output = dataPro_m3d_code(relation, entityLink, video_id, "en") 
        else:
            prompt_1 = """def information_extraction(input_text, scene_graph, input_image):
            \"\"\"
                第一，从文本中提取实体。
                第二，基于实体提取实体链。
                第三，基于实体链提取实体链关系。
                第四，推断图像中的视觉区域坐标和类型。
            \"\"\"
            input_text = \"""" + text + """\"
            entity_attribute = """ + json.dumps(attr["zh"][video_id] if video_id in attr["zh"] else {}, ensure_ascii=False) + """
            scene_graph = """ + json.dumps({f"Img{i}": item[-1] for i, item in enumerate(sorted(graph["zh"][video_id].items(), key=lambda x: int(x[0].split('.')[0])))} if video_id in graph["zh"] else {}, ensure_ascii=False) + """ 
            input_image = """ 
            prompt_2 = """\nentity_dic = {}
                chain_dic = {}
                relation_dic = {}
                grounding_dic = {}
                # 抽取实体，实体链，关系和视觉区域"""

            output = dataPro_m3d_code(relation, entityLink, video_id, "zh") 

        if name == "train":
            src = prompt_1.strip()
            result = prompt_2 + "\n" + output.strip() + " </s>"
        else:
            src = prompt_1.strip()
            result = prompt_2 + "\n"
        # print(src)
        # print(result)
        # exit()
        if video_id != "none": 
            video_atts = torch.load(f"{video_path}/{video_id}_frame_atts.pth").to('cpu').long()
            video_hidden = torch.load(f"{video_path}/{video_id}_frame_hidden_state.pth").to('cpu').float()
            video_query = torch.load(f"{video_path}/{video_id}_video_query_tokens.pth").to('cpu').float()
        else:
            video_atts = torch.zeros((1, 1)).long()
            video_hidden = torch.zeros((1, 1, 768)).float()
            video_query = torch.zeros((1, 1, 768)).float()
        video_atts_list.append(video_atts)
        video_hidden_list.append(video_hidden)
        video_query_list.append(video_query)
        test_list.append(src)
        src_item = tokenizer(src)
        tgt_item = tokenizer(result)
        src_ids = src_item.input_ids
        tgt_ids = tgt_item.input_ids
        src_mask = src_item.attention_mask
        tgt_mask = tgt_item.attention_mask
        src_len = src_ids.__len__()
        tgt_len = tgt_ids.__len__()
        if src_len + tgt_len > data_max_len:
            data_max_len = src_len + tgt_len
        src_list.append(src_ids)
        src_mask_list.append(src_mask)
        tgt_list.append(tgt_ids)
        tgt_mask_list.append(tgt_mask)
        link_list.append(entityLink)
        relation_list.append(relation)
        video_id_list.append(video_id)
        docs.append(text)
        vg_list.append("")
    print(data_max_len)
    return src_list, src_mask_list, tgt_list, tgt_mask_list, video_atts_list, video_hidden_list, video_query_list, vg_list, relation_list, link_list, test_list, video_id_list, docs

def load_data_bert(tokenizer, myconfig, model):
    lang = myconfig.dataset.split("_")[-1]
    with open(f'./data/m3d/{lang}/train_{lang}.json', 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())
    with open(f'./data/m3d/{lang}/test_{lang}.json', 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())

    train_dataset = MyDataset(*process_bert(train_data, tokenizer, myconfig, name="train"))
    test_dataset = MyDataset(*process_bert(test_data, tokenizer, myconfig, name="test"))
    return train_dataset, test_dataset