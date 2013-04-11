all: _music.so

clean:
	rm -r _music.so _music.c

distclean:
	git clean -fd

check: all
	python _tests.py check

profile: all
	python _tests.py profile

_music.so: _music.pyx setup.py
	python setup.py build_ext --inplace

