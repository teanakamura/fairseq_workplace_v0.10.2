ks=(0 1 2 3 4 5)
# ps=(010 020 030 050)
ps=(010)
# bases=(task1_all task2_keyword_set1 task2_keyword_set2 task2_keyword_set3 task2_keyword_set_all)
# bases=(task1_all task2_keyword_set_all)
bases=(task1_all task2_keyword_set1 task2_keyword_set2 task2_keyword_set3 task2_keyword_set_all task1_certain0 task1_certain1 task1_certain2 task2_keyword_set_certain0 task2_keyword_set_certain1 task2_keyword_set_certain2)

for p in ${ps[@]}; do
  # source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}_inline/base/css_srcannt${p}_inline_unigram16000.yml

  for k in ${ks[@]}; do
    # source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}/base/test_css_srcannt_certain_number${k}.yml
  #   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}/con/test_css_srcannt_certain_number${k}.yml
  #   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}/sum/test_css_srcannt_certain_number${k}.yml
  #   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}_inline/base/test_css_srcannt_certain_number${k}.yml
  #   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}_inline/con/test_css_srcannt_certain_number${k}.yml
  #   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}_inline/sum/test_css_srcannt_certain_number${k}.yml
  done

  for base in ${bases[@]}; do
    # source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}/base/test_baobab_${base}_10best.yml
    source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt${p}/base/test_baobab_${base}.yml
  done
done

# for k in ${ks[@]}; do
#   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt_inline_synthetic/base/test_css_srcannt_certain_number${k}.yml
# done

# for base in ${bases[@]}; do
#   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt_inline_synthetic/base/test_baobab_${base}.yml
# done


# for k in ${ks[@]}; do
#   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt_inline_synthetic/base/test_css_srcannt_certain_number${k}.yml
# done
#
# for base in ${bases[@]}; do
#   source qsub_yaml.sh -d jijinews -e tf_generate_yaml.sh -c css_srcannt_inline_synthetic/base/test_baobab_${base}.yml
# done
