import sentencepiece as spm
import os
import argparse
import yaml
from attrdict import AttrDict

from py_utils.load_yaml import load_yaml, hierarchical_load_yaml

HOME=os.environ['HOME']
SPMODEL_ROOT=f'{HOME}/data'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', default='../script/yaml_configs/env.yml')
    parser.add_argument('--yaml-root', default='../script/yaml_configs')
    parser.add_argument('-y', '--yaml', required=True)
    parser.add_argument('--only-system', action='store_true')
    parser.add_argument('--only-reference', action='store_true')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--remove-id', action='store_true')
    return parser.parse_args()

def postprocess_enclosure(conf):
    if conf.get('pstp') and conf.pstp.get('sp_model'):
        sp = spm.SentencePieceProcessor()
        sp.load(conf.pstp.sp_model)

    def postprocess(fl):
        print(f'FILE: {conf.pstp.path}/{fl}')
        with open(f'{conf.pstp.path}/{fl}') as f:
            lines = f.readlines()

        if conf.pstp.sort:
            print(f'SORT BY ID')
            sorted_lines = [''] * len(lines)
            for i, l in enumerate(lines, 1):
                print(f'{i}/{len(lines)}', end='\r')
                idx, snt = l.split(None, 1)
                sorted_lines[int(idx)] = snt
            lines = sorted_lines

        ids, lines = zip(*[l.split(None, 1) for l in lines])

        tmp_lines = []
        for i, l in enumerate(lines, 1):
            print(f'{i}/{len(lines)}', end='\r')
            pieces = l.strip().split(None)
            if conf.get('pstp') and conf.pstp.get('sp_model'):
                tmp_lines.append(sp.DecodePieces(pieces))
            elif conf.get('pstp') and conf.pstp.get('simple'):
                tmp_lines.append(''.join(pieces))
        lines = tmp_lines

        if conf.pstp.remove_tokens:
            print(f'REMOVE TOKENS: {" ".join(conf.pstp.remove_tokens)}')
            tmp_lines = []
            for l in lines:
                for rmtkn in conf.pstp.remove_tokens:
                    l = l.replace(rmtkn, '')
                tmp_lines.append(l)
            lines = tmp_lines
        return ids, lines

    return postprocess

def conf_pstp(conf):
    ret = dict()

    pstp_path = f'{env.FAIRSEQ_ROOT}/workplace/generation/{conf.data.name}/{conf.data.type}/{conf.model.name}{conf.checkpoint}'
    if conf.get('pstp') and conf.pstp.get('data'):
        pass  ## 現在どこでも使っていないので今後必要に応じて拡張
    elif conf.get('gen') and conf.gen.get('data'):
        pstp_path = os.path.join(pstp_path, conf.gen.data)
        if conf.gen.get('out'):
            pstp_path = os.path.join(pstp_path, conf.gen.out)

    if conf.get('sp_model'):
        ret['sp_model'] = f'{SPMODEL_ROOT}/{conf.data.name}/{conf.sp_model}'
    else:
        ret['simple'] = True

    if args.only_system:
        pstp_files = ['system_output.txt']
    elif args.only_reference:
        pstp_files = ['reference.txt']
    else:
        pstp_files = conf.pstp.files if conf.get('pstp') and conf.pstp.get('files') else ['system_output.txt', 'reference.txt']

    pstp_sort = conf.pstp.sort if conf.get('pstp') and conf.pstp.get('sort') else False
    if args.sort: pstp_sort = True

    pstp_rmtokens = conf.pstp.remove_tokens if conf.get('pstp') and conf.pstp.get('remove_tokens') else False

    pstp_rmid = conf.pstp.remove_id if conf.get('pstp') and conf.pstp.get('remove_id') else False
    if args.remove_id: pstp_rmid = True

    ret.update(dict(
                path = pstp_path,
                files = pstp_files,
                sort = pstp_sort,
                remove_tokens = pstp_rmtokens,
                remove_id = pstp_rmid,
                ext = 'dtk'
              ))
    return ret


if __name__ == '__main__':
    args = parse()
    env = load_yaml(__file__, args.env)
    conf = hierarchical_load_yaml(
             __file__,
             args.yaml_root,
             args.yaml,
             verbose=True,
           )

    conf.update({'pstp': conf_pstp(conf)})
    from pprint import pprint as pp
    pp(conf)

    postprocess = postprocess_enclosure(conf)
    for fl in conf.pstp.files:
        ids, lines = postprocess(fl)
        with open(f'{conf.pstp.path}/{fl}.{conf.pstp.ext}', 'w') as f:
            for idx, l in zip(ids, lines):
                if conf.pstp.remove_id:
                    f.write(l + '\n')
                else:
                    f.write(f'{idx} {l}\n')
