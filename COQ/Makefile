COQC = coqc
COQDEP = coqdep
COQFLAGS = -Q . GIFT

VFILES = $(shell cat _CoqProject | grep '\.v$$')
VOFILES = $(VFILES:.v=.vo)

all: $(VOFILES)

%.vo: %.v
	$(COQC) $(COQFLAGS) $<

depend:
	$(COQDEP) $(COQFLAGS) $(VFILES) > .depend

clean:
	rm -f $(VOFILES) $(VFILES:.v=.glob) $(VFILES:.v=.vok) $(VFILES:.v=.vos) .depend

.PHONY: all depend clean

-include .depend
