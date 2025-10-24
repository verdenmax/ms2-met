.PHONY: clean run

run:
	@toilet -f mono12 "start" | lolcat
	@python3 main.py
	@toilet -f mono12 "end" | lolcat

clean:
	rm -f checkpoint.pkl
	rm -f c13_mix.csv
	rm -f *.cache
	rm -f *fake.mgf
