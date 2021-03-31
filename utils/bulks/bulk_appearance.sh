source ~/venv/pytorch/bin/activate
ps=(010 020 030 050)
ks=(0 1 2 3 4 5)
bases=(task1_free1 task1_free2 task1_free3 task1_all task2_keyword_set1 task2_keyword_set2 task2_keyword_set3 task2_keyword_set_all)

for p in ${ps[@]}; do
  # python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}_inline/base/css_srcannt${p}_inline_unigram16000.yml

  for k in ${ks[@]}; do
    # python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}/con/test_css_srcannt_certain_number${k}.yml
    # python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}/sum/test_css_srcannt_certain_number${k}.yml
    python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}_inline/base/test_css_srcannt_certain_number${k}.yml
    # python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}_inline/con/test_css_srcannt_certain_number${k}.yml
    # python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}_inline/sum/test_css_srcannt_certain_number${k}.yml
  done

  for base in ${bases[@]}; do
    python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt${p}_inline/base/test_baobab_${base}.yml
  done
done

# python calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt_inline_synthetic/base/css_srcannt_inline_synthetic.yml
for k in ${ks[@]}; do
  python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt_inline_synthetic/base/test_css_srcannt_certain_number${k}.yml
done

for base in ${bases[@]}; do
  python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt_inline_synthetic/base/test_baobab_${base}.yml
done

# python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt010/base/css_srcannt010_unigram16000.yml
# for k in ${ks[@]}; do
#   python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt010/base/test_css_srcannt_certain_number${k}.yml
# done
# for base in ${bases[@]}; do
#   python ../calc_keyword_appearance_rate_yaml.py -d jijinews -y css_srcannt010/base/test_baobab_${base}.yml
# done

