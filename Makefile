all: _music.so

clean:
	rm -r _music.so _music.c

distclean:
	git clean -fd

check: all
	python _tests.py check

profile: all
	python _tests.py profile
	nohup runsnake "spectrum.gprofile" 2>&1 >/dev/null &

spectrum: all
	python _tests.py spectrum

_music.so: _music.pyx setup.py cmusic.c cmusic.h
	python setup.py build_ext --inplace

