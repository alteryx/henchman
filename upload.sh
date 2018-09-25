#!/bin/bash

versions="linux-64 win-64 osx-64"
make clean
conda build .
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/* -r pypi
for python_henchman in $(conda build . --output); do
    for version in $versions; do
        conda convert --platform "$version" "$python_henchman" -o outputdir/ &&
        mkdir -p outputdir/osx-64 &&
        cp "$python_henchman" outputdir/osx-64/
    done
    for other_arch in outputdir/*/$(basename "$python_henchman"); do
        anaconda upload --user featurelabs "$other_arch"
    done
done
