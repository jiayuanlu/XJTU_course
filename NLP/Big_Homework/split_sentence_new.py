import re

def cut_zh_en(para, thres=20):
    para = re.sub('([，。！？；])([^：])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^：])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^：])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([：:][“‘])([^：:])', r'\1\n\2', para)
    para = re.sub('“|‘|”|’', '', para)
    para = re.sub('（[一二三四五六七八九十]+）|[一二三四五六七八九十]+、|\([一二三四五六七八九十]+\)', '', para) # 删除序号
    para = re.sub('\n+', '\n', para) # 删除多余的换行符
    para = re.sub('　|——', '', para) # 删除中文的宽空格、连字符
    para = para.rstrip()  # 删除段尾的换行符
    texts = para.split("\n")

    idx = 0
    while idx < len(texts):
        if len(texts[idx]) < thres and idx != len(texts) - 1:
            texts[idx] = texts[idx] + texts[idx + 1]
            del texts[idx + 1]
        else:
            idx += 1
    return texts


if __name__ == "__main__":
    with open("comp//3.txt") as file:
        text = file.read()
        text = cut_zh_en(text)
        print("\n".join(text[:]))