import argparse
import numpy as np
from sumeval.metrics.rouge import RougeCalculator
import contextlib
import spacy
from collections import defaultdict
#from utils import process_bpe_symbol
import os
import sys
import yaml
from attrdict import AttrDict


FAIRSEQ_ROOT=f'{os.environ["HOME"]}/fairseq'
USER_DIR='/fs1/groups1/gcb50243/nakamura'
DATA_ROOT=f'{os.environ["HOME"]}/data'

print(FAIRSEQ_ROOT)
print(USER_DIR)

def deepupdate(dict_base, other):
  for k, v in other.items():
    if isinstance(v, dict) and k in dict_base:
      deepupdate(dict_base[k], v)
    else:
      dict_base[k] = v

def read_file(filename, bpe_symbol=None):
  with open(filename) as f:
    # lines = [process_bpe_symbol(line.strip(), bpe_symbol) for line in f]
    # lines = [l.split() for l in f]
    return f.readlines()

@contextlib.contextmanager
def dummy_context_mgr():
    yield dummy_generator()
def dummy_generator():
    while True:
        yield None

class KeywordRemover():
    def __init__(self, keypath):
        self.sp = spacy.load('en')
        with open(keypath) as f:
            self.keys = f.readlines()
    def __call__(self, doc):
        idx, doc = doc.split(None, 1)
        sp_doc = self.sp(doc.rstrip())
        keys = self.keys[int(idx)]
        keys = keys.rstrip()
        first_word_dict = defaultdict(list)
        for vk in keys.split(', '):
            v, *ks = vk.split()
            if float(v) < 0.05:
                break
            first_word_dict[ks[0]].append(ks)

        it = iter(sp_doc)
        tmp = []
        res = []
        while True:
            try:
                token = next(it) if not tmp else tmp.pop()
            except StopIteration:
                break
            if token.lemma_ in first_word_dict:
                for key in first_word_dict[token.lemma_]:
                    try:
                        for _ in range(2-len(tmp)):
                            tmp.insert(0, next(it))  # [nntoken, ntoken]
                    except StopIteration:
                        pass
                    if len(tmp) >= len(key)-1:
                        for i in range(1, len(key)):
                            if not tmp[-i].lemma_ == key[i]:
                                break
                        else:
                            for _ in range(len(key)-1):
                                tmp.pop()
                            break
                else:
                    res.append(token.text)
            else:
                res.append(token.text)
        return ' '.join(res)

def main(conf):
    id_list = []
    rougeone_list = []
    rougetwo_list = []
    rougel_list = []
    rouge4one = RougeCalculator(stopwords=True, lang=conf.rouge.lang)
    rouge4other = RougeCalculator(stopwords=False, lang=conf.rouge.lang)
    keynum_counter = defaultdict(lambda: {'count': 0, 'rouge1_lis': [], 'rouge2_lis': [], 'rougel_lis': []})
    if conf.rouge.get('keyword', None):
        kr = KeywordRemover(conf.rouge.keyword)
    with open(conf.system_out) as sf, \
         open(conf.reference) as rf, \
         open(conf.test_src) as tsrcf:
        assert len(sf.readlines()) == len(rf.readlines()) == len(tsrcf.readlines())
    with open(conf.system_out) as sf, \
         open(conf.reference) as rf, \
         open(conf.test_src) as tsrcf:
        tsrcs = tsrcf.readlines()
        for i, (so, re) in enumerate(zip(sf, rf)):
            print(i, end='\r', flush=True)
            if conf.rouge.get('keyword', None):
                so = kr(so)
                re = kr(re)
            else:
                idx, so = so.split(None, 1)
                idx, re = re.split(None, 1)
            id_list.append(int(idx))
            rouge1 = rouge4one.rouge_1(summary=so, references=re, alpha=conf.alpha)
            rouge2 = rouge4other.rouge_2(summary=so, references=re, alpha=conf.alpha)
            rougel = rouge4one.rouge_l(summary=so, references=re, alpha=conf.alpha)
            rougeone_list.append(rouge1)
            rougetwo_list.append(rouge2)
            rougel_list.append(rougel)
            keynum = tsrcs[int(idx)].count('</@>')
            keynum_counter[keynum]['count'] += 1
            keynum_counter[keynum]['rouge1_lis'].append(rouge1)
            keynum_counter[keynum]['rouge2_lis'].append(rouge2)
            keynum_counter[keynum]['rougel_lis'].append(rougel)
    lowest_idids = np.argpartition(rougeone_list, 100)[:100]
    print(f"Lowest IDs\t{' '.join(map(str, np.array(id_list)[lowest_idids]))}")
    print('ROUGE-1\t%.6f'%(np.average(rougeone_list)))
    print('ROUGE-2\t%.6f'%(np.average(rougetwo_list)))
    print('ROUGE-L\t%.6f'%(np.average(rougel_list)))
    if conf.output:
        with open(conf.output, 'w') as of:
            for idx, r1, r2, rl in zip(id_list, rougeone_list, rougetwo_list, rougel_list):
                of.write(f'{idx}, {r1}, {r2}, {rl}\n')

    for keynum in range(5):
        print(keynum)
        print(f'count: {keynum_counter[keynum]["count"]}')
        print('ROUGE-1\t%.6f'%(np.average(keynum_counter[keynum]['rouge1_lis'])))
        print('ROUGE-2\t%.6f'%(np.average(keynum_counter[keynum]['rouge2_lis'])))
        print('ROUGE-L\t%.6f'%(np.average(keynum_counter[keynum]['rougel_lis'])))
  
   # system_out_list = read_file(args.system_out, args.remove_bpe)
   # reference_list = read_file(args.reference, args.remove_bpe)
   # rougetwo_list = []
   # rougel_list = []
   # for index, snt in enumerate(system_out_list):
   # rougeone_list.append(rouge4one.rouge_1(summary=snt, references=reference_list[index], alpha=args.alpha))
   # rougetwo_list.append(rouge4other.rouge_2(summary=snt, references=reference_list[index], alpha=args.alpha))
   # rougel_list.append(rouge4one.rouge_l(summary=snt, references=reference_list[index], alpha=args.alpha))
   # print('ROUGE-1\t%.6f'%(np.average(rougeone_list)))
   # print('ROUGE-2\t%.6f'%(np.average(rougetwo_list)))
   # print('ROUGE-L\t%.6f'%(np.average(rougel_list)))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', dest='system_out',
        default=None,
        help='specify the system output file name')
    parser.add_argument('-r', '--reference', dest='reference',
        default=None,
        help='specify the reference file name')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-l', '--lang', default='en')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--remove-bpe', default=None)
    parser.add_argument('-y', '--yaml', required=True)
    parser.add_argument('-d', '--data', required=True)
    return parser.parse_args()

def load_yaml(args):
    yaml_root_path = f'{FAIRSEQ_ROOT}/workplace/script/yaml_configs'
    conf = yaml.safe_load(open(f'{yaml_root_path}/default.yml'))
    if os.path.isfile(f'{yaml_root_path}/{args.data}/default.yml'):
        deepupdate(conf, yaml.safe_load(open(f'{yaml_root_path}/{args.data}/default.yml')))
    deepupdate(conf, yaml.safe_load(open(f'{yaml_root_path}/{args.data}/{args.yaml}')))
    return AttrDict(conf)

if __name__ == "__main__":
    args = parse()
    conf = load_yaml(args)
    gen_path = f'{FAIRSEQ_ROOT}/workplace/generation/{conf.data.name}/{conf.data.type}/{conf.model.name}{conf.checkpoint}'
    print(gen_path)
    if conf.get('gen_data'):
        gen_path += f'/{conf.get("gen_data")}'
    conf.system_out = args.system_out or f'{gen_path}/system_output.{conf.rouge.suf}'
    conf.reference = args.reference or f'{gen_path}/reference.{conf.rouge.suf}'
    conf.test_src = f'{DATA_ROOT}/{conf.data.name}/{conf.rouge.test.src}'
    conf.output = conf.rouge.get('output') or args.output   #or f'{gen_path}/rougeout.txt'
    conf.alpha = conf.rouge.get('alpha') or args.alpha or 0.5
    main(conf)
