import os
LTP_DATA_DIR = os.path.join(os.getcwd(), 'ltp_model')
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')

from pyltp import Parser, Segmentor, Postagger
parser = Parser()
segmentor = Segmentor()
postagger = Postagger()


parser.load(par_model_path)
segmentor.load(cws_model_path)
postagger.load(pos_model_path)

sent = "丽丽派遣小明去求援"
words = segmentor.segment(sent)
postags = postagger.postag(words)
arcs = parser.parse(words, postags)

print(list(enumerate(words)))

print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

tmp = []
for index, arc in enumerate(arcs):
    if arc.relation in ['VOB', 'SBV']:
        tmp.append(words[index])
        tmp.append(words[arc.head-1])
    elif arc.relation == 'HED':
        tmp.append(words[index])

for index, tag in enumerate(postags):
    if tag == 'v':
        tmp.append(words[index])

tmp = set(tmp) - {'什么', '谁', '哪儿', '哪里'}
print(set(tmp))

parser.release()
segmentor.release()
postagger.release()
