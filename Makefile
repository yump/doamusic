build: _music.so

clean:
	rm -r _music.so _music.c

distclean:
	git clean -fd

debug:
	python setup.py build_ext --inplace -D DEBUG

check: build
	python _tests.py check

profile: build
	python _tests.py profile
	nohup runsnake "spectrum.gprofile" 2>&1 >/dev/null &
	nohup runsnake "doasearch.gprofile" 2>&1 >/dev/null &

spectrum: build
	python _tests.py spectrum

timetrial: build
	python _tests.py timetrial

doasearch: build
	python _tests.py doasearch

_music.so: _music.pyx setup.py cmusic.c cmusic.h
	python setup.py build_ext --inplace

