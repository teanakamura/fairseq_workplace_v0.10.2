import spacy
import argparse
import numpy as np
import contextlib
import spacy
from collections import defaultdict
import os
import sys
import yaml
from attrdict import AttrDict

FAIRSEQ_ROOT=f'{os.environ["HOME"]}/fairseq'
USER_DIR='/fs1/groups1/gcb50243/nakamura'
DATA_ROOT=f'{os.environ["HOME"]}/data'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', dest='target',
                        help='target file name')
    parser.add_argument('-k', '--keyword', dest='keyword',
                        help='keyword file name')
    parser.add_argument('-d', '--data', dest='data',
                        help='data name')
    parser.add_argument('-y', '--yaml', dest='yaml',
                        help='config file name')
    parser.add_argument('--threshold', default=0.05)
    return parser.parse_args()

def deepupdate(dict_base, other):
    for k, v in other.items():
        if isinstance(v, dict) and k in dict_base:
            deepupdate(dict_base[k], v)
        else:
            dict_base[k] = v

def read_config(config, conf_type, data=''):
    if conf_type == 'conf':
        confdir = f'{FAIRSEQ_ROOT}/workplace/script/configs'
        conf = dict()
        with open(f'{confdir}/{config}') as f:
            for l in f:
                l = l.strip()
                if l:
                    k, v = l.split('=')
                    conf[k] = v
    elif conf_type == 'yaml':
        yamldir = f'{FAIRSEQ_ROOT}/workplace/script/yaml_configs'
        conf = yaml.safe_load(open(f'{yamldir}/default.yml'))
        conf_add = yaml.safe_load(open(f'{yamldir}/{data}/default.yml'))
        deepupdate(conf, conf_add)
        conf_add = yaml.safe_load(open(f'{yamldir}/{data}/{config}'))
        deepupdate(conf, conf_add)
    print(conf)
    return conf

def read_file(filename, bpe_symbol=None):
  with open(filename) as f:
    # lines = [process_bpe_symbol(line.strip(), bpe_symbol) for line in f]
    # lines = [l.split() for l in f]
    return f.readlines()

def enclosure(funcname, threshold=None):
    sp = spacy.load('en')
    def parse_keyword_line(l):
        for k in l.rstrip().split(','):
            yield(k)
    def parse_keyword_value_line(l):
        for vk in l.rstrip().split(', '):
            v, ks = vk.split(None, 1)
            if float(v) > threshold:
                yield([t.lemma_ for t in sp(ks)])
    def parse_sentence(l):
        return [t.lemma_ for t in sp(l)]
    if funcname == 'parse_keyword_line':
        return parse_keyword_line
    elif funcname == 'parse_keyword_value_line':
        return parse_keyword_value_line
    elif funcname == 'parse_sentence':
        return parse_sentence

def main(target_path, keyword_path, threshold):
    with open(keyword_path) as f:
        parse_keyword_line = enclosure('parse_keyword_line', threshold)
        keywords = list(map(parse_keyword_line, f.readlines()))
    with open(target_path) as f:
        parse_sentence = enclosure('parse_sentence')
        rate = 0
        num_val_lines = 0
        for i, l in enumerate(f, 1):
            # print(f'{i}/{len(keywords)}', end='\r')
            idx, sent = l.split(None, 1)
            keyword_set = list(keywords[int(idx)])
            if not keyword_set: continue
            rate_add = 0
            for j, keyword in enumerate(keyword_set, 1):
                if not keyword: break
                # parsed_sent = parse_sentence(sent)
                # binary = ' '.join(keyword) in ' '.join(parsed_sent)
                # if not binary:
                #     print(idx)
                #     print(parsed_sent)
                #     print(keyword)
                #     print()
                # rate_add += binary
                # rate_add += ' '.join(keyword) in ' '.join(parse_sentence(sent))
                rate_add += keyword in sent
            else:
                rate += rate_add / j
                num_val_lines += 1
        print(i, rate, num_val_lines, rate / max(num_val_lines, 1))

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

if __name__ == '__main__':
    args = parse()
    conf = load_yaml(args)

    gen_path = f'{FAIRSEQ_ROOT}/workplace/generation/{conf.data.name}/{conf.data.type}/{conf.model.name}{conf.checkpoint}'
    if conf.get('gen') and conf.gen.get('data'):
        gen_path = os.path.join(gen_path, conf.gen.data)
    args.target = args.target or f'{gen_path}/system_output.{conf.rouge.suf}'
    args.keyword = args.keyword or f'{DATA_ROOT}/{conf.data.name}/{conf.key}'
    # print(args.target)
    # print(args.keyword)
    main(args.target, args.keyword, args.threshold)


# sp = spacy.load('en')
# tokens = sp('i don\'t like you.')
# for t in tokens:
#     print(t.lemma_)
