#!/bin/bash

<<HOWTOUSE
./load_yaml.sh <yaml_root_path> <yaml_path_rel_to_yaml_root_path>
yaml_root_pathから目的のyamlファイルへの相対パスであるyaml_path_rel_to_yalm_root_path末端までの各ディレクトリに関してdefault.ymlがあったらLOADED_YAMLに読み込んで，より深い層のdefault.ymlファイルで上書きしていく．最後にyaml_root_path/yaml_pathのファイルを読み込んでLOADED_YAMLを上書きする
HOWTOUSE

update_conf () {
  while read line
  do
    if echo $line | grep -F = &>/dev/null
    then
      varname=$(echo "$line" | cut -d '=' -f 1)
      LOADED_YAML[$varname]=$(echo "$line" | cut -d '=' -f 2- | sed -e 's/^"//' -e 's/"$//')
    fi
  done < <(parse_yaml $1)
}

yaml_root_path=$1
yaml_path=$2

yaml_path_array=(${yaml_path//\// })
declare -A LOADED_YAML
for e in ${yaml_path_array[@]}; do
  if [ -f $yaml_root_path/default.yml ]; then
    update_conf $yaml_root_path/default.yml
  fi
  yaml_root_path=$yaml_root_path/$e
done
update_conf $yaml_root_path

for k in ${!LOADED_YAML[@]}; do
  echo [$k]=${LOADED_YAML[$k]}
done
