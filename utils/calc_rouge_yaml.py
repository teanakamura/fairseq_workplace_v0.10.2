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

# def read_config(config):
#     conf = dict()
#     with open(f'{FAIRSEQ_ROOT}/workplace/script/configs/{config}') as f:
#         for l in f:
#             l = l.strip()
#             if l:
#                 k, v = l.split('=')
#                 conf[k] = v
#     print(conf)
#     return conf

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
        # self.sp = spacy.load('en')
        with open(keypath) as f:
            self.keys = [keyline.strip().split(',') for keyline in f.readlines()]
    def __call__(self, idx, txt):
        keys = self.keys[idx]
        for key in keys:
            txt = txt.replace(key, '')
        return txt
    # def __call__(self, doc):
    #     idx, doc = doc.split(None, 1)
    #     sp_doc = self.sp(doc.rstrip())
    #     keys = self.keys[int(idx)]
    #     keys = keys.rstrip()
    #     first_word_dict = defaultdict(list)
    #     for vk in keys.split(', '):
    #         v, *ks = vk.split()
    #         if float(v) < 0.05:
    #             break
    #         first_word_dict[ks[0]].append(ks)
    #
    #     it = iter(sp_doc)
    #     tmp = []
    #     res = []
    #     while True:
    #         try:
    #             token = next(it) if not tmp else tmp.pop()
    #         except StopIteration:
    #             break
    #         if token.lemma_ in first_word_dict:
    #             for key in first_word_dict[token.lemma_]:
    #                 try:
    #                     for _ in range(2-len(tmp)):
    #                         tmp.insert(0, next(it))  # [nntoken, ntoken]
    #                 except StopIteration:
    #                     pass
    #                 if len(tmp) >= len(key)-1:
    #                     for i in range(1, len(key)):
    #                         if not tmp[-i].lemma_ == key[i]:
    #                             break
    #                     else:
    #                         for _ in range(len(key)-1):
    #                             tmp.pop()
    #                         break
    #             else:
    #                 res.append(token.text)
    #         else:
    #             res.append(token.text)
    #     return ' '.join(res)

def main(args):
    id_list = []
    rougeone_list = []
    rougetwo_list = []
    rougel_list = []
    rouge4one = RougeCalculator(stopwords=True, lang=args.lang)
    rouge4other = RougeCalculator(stopwords=False, lang=args.lang)
    if args.keyword:
        kr = KeywordRemover(args.keyword)
    print(args.system_out)
    print(args.reference)
    with open(args.system_out) as sf, \
         open(args.reference) as rf:
        for i, (so, re) in enumerate(zip(sf, rf)):
            # if i > 10: break
            if args.progress:
                print(i, end='\r', flush=True)
            try:
                idx, so = so.split(None, 1)
                idx, re = re.split(None, 1)
                idx = int(idx)
            except ValueError:
                if args.keyword:
                    raise
                so = so.strip()
                re = re.strip()
            if args.keyword:
                so = kr(idx, so)
                re = kr(idx, re)
            # id_list.append(int(idx))
            rougeone_list.append(rouge4one.rouge_1(summary=so, references=re, alpha=args.alpha))
            rougetwo_list.append(rouge4other.rouge_2(summary=so, references=re, alpha=args.alpha))
            rougel_list.append(rouge4one.rouge_l(summary=so, references=re, alpha=args.alpha))
    # lowest_idids = np.argpartition(rougeone_list, 100)[:100]
    # print(f"Lowest IDs\t{' '.join(map(str, np.array(id_list)[lowest_idids]))}")
    print('ROUGE-1\t%.6f'%(np.average(rougeone_list)))
    print('ROUGE-2\t%.6f'%(np.average(rougetwo_list)))
    print('ROUGE-L\t%.6f'%(np.average(rougel_list)))
    # with open(args.output, 'w') as of:
    #     for idx, r1, r2, rl in zip(id_list, rougeone_list, rougetwo_list, rougel_list):
    #         of.write(f'{idx}, {r1}, {r2}, {rl}\n')
  
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
    parser.add_argument('-l', '--lang')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--remove-bpe', default=None)
    parser.add_argument('-y', '--yaml')
    parser.add_argument('-k', '--keyword', default=None)
    parser.add_argument('-d', '--data')
    parser.add_argument('-p', '--progress', action='store_true')
    return parser.parse_args()

def load_yaml(args):
    yaml_root_path = f'{FAIRSEQ_ROOT}/workplace/script/yaml_configs'
    conf = yaml.safe_load(open(f'{yaml_root_path}/default.yml'))
    tmp_path = yaml_root_path
    subpaths = [args.data] + args.yaml.split('/')
    print(subpaths)
    for subpath in subpaths:
        if os.path.isfile(f'{tmp_path}/default.yml'):
            # print(f'LOAD: {tmp_path}/default.yml')
            deepupdate(conf, yaml.safe_load(open(f'{tmp_path}/default.yml')))
        tmp_path = os.path.join(tmp_path, subpath)
    print(f'LOAD: {tmp_path}')
    deepupdate(conf, yaml.safe_load(open(tmp_path)))
    return AttrDict(conf)

if __name__ == "__main__":
    args = parse()
    if args.yaml:
        conf = load_yaml(args)
        # args = parse()
        # conf_path = f'{FAIRSEQ_ROOT}/workplace/script/yaml_configs/default.yml'
        # conf = yaml.safe_load(open(conf_path))
        # conf_path = f'{FAIRSEQ_ROOT}/workplace/script/yaml_configs/{args.data}/default.yml'
        # conf_add = yaml.safe_load(open(conf_path))
        # deepupdate(conf, conf_add)
        # conf_path = f'{FAIRSEQ_ROOT}/workplace/script/yaml_configs/{args.data}/{args.config}'
        # conf_add = yaml.safe_load(open(conf_path))
        # deepupdate(conf, conf_add)

        # data = conf.get('gen_data') or conf['data']
        # gen_path = f'{FAIRSEQ_ROOT}/workplace/generation/{data}/{conf["model"]}{conf["checkpoint"]}'
        gen_path = f'{FAIRSEQ_ROOT}/workplace/generation/{conf.data.name}/{conf.data.type}/{conf.model.name}{conf.checkpoint}'
        if conf.get('gen') and conf.gen.get('data'):
            gen_path = os.path.join(gen_path, conf.gen.data)
        # print(gen_path)
    if args.keyword == 'yaml':
        args.keyword = f'{DATA_ROOT}/{conf.data.name}/{conf.key}'
    args.system_out = args.system_out or f'{gen_path}/system_output.{conf.rouge.suf}'
    args.reference = args.reference or f'{gen_path}/reference.{conf.rouge.suf}'
    # args.output = args.output or f'{gen_path}/rougeout.txt'
    args.lang = args.lang or conf.rouge.lang
    if not args.lang:
        raise
    main(args)
