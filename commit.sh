#/bin/bash

# 
if [ "$1" = "" ];then
conda install plotly
conda install pandas
conda install imageio
fi

# regenerate .whl
echo "re-build .whl file ....................................."
echo

rm -rf build dist MPUtils.egg-info
python setup.py bdist_wheel --universal

echo
echo "re-build .whl file complete."
echo

# git update
if [ "$1" == "git" ]
then
echo "github update .........................................."
echo

git add -A
git commit -m "normal update"
git push

echo 
echo "github update complete."
echo
fi

#
echo "re-install MPUtils ....................................."
echo

cd ..
pip uninstall MPUtils
pip install MPUtils/dist/MPUtils-0.1.0-py2.py3-none-any.whl
cd -

echo
echo "re-install MPUtils complete."
echo

echo "all done."
