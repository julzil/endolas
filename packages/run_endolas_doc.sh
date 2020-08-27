sphinx-apidoc -M -o endolas_doc/rsts endolas --ext-todo --force
rm endolas_doc/rsts/modules.rst
cd endolas_doc
make html
cd ..
rm endolas_doc.html
ln -s endolas_doc/_build/html/index.html endolas_doc.html

