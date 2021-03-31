ps=(010 020 030 050)
ks=(0 1 2 3 4 5)
for p in ${ps[@]}; do
  for k in ${ks[@]}; do
    python sort_generation.py -d jijinews -y css_srcannt${p}/con/test_css_srcannt_certain_number${k}.yml
    python sort_generation.py -d jijinews -y css_srcannt${p}/sum/test_css_srcannt_certain_number${k}.yml
    python sort_generation.py -d jijinews -y css_srcannt${p}_inline/base/test_css_srcannt_certain_number${k}.yml
    python sort_generation.py -d jijinews -y css_srcannt${p}_inline/con/test_css_srcannt_certain_number${k}.yml
    python sort_generation.py -d jijinews -y css_srcannt${p}_inline/sum/test_css_srcannt_certain_number${k}.yml
  done
done
