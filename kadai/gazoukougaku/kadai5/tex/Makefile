EDITOR		= emacs

HEADER		= report

MAINTEX		= $(HEADER).tex

TEX_FILE	= $(HEADER).tex

MAINDVI		= $(HEADER).dvi

SUFFIX		= .tex:s

SRCS		= $(HEADER).tex

$(MAINDVI):	$(SRCS) $(TEX_FILE)
		platex $(MAINTEX)
		platex $(MAINTEX)

force:;		platex $(MAINTEX)

pdf:		$(MAINDVI)
		dvipdfmx -o $(HEADER).pdf $(MAINDVI)
		evince $(HEADER).pdf &

edit:;		$(EDITOR) $(MAINTEX) &

clean:;		rm -f *.aux *.log *.out *.dvi *.pdf  *~
