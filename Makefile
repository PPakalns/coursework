default: none

CAT_DIR := ../CAT
GEN_DIR := ../GEN

TO_DIR := ../TO
TS_DIR := ../TS

DATATO := ../to.npy
DATATS := ../ts.npy

JPGS := $(wildcard ${CAT_DIR}/*.jpg)
GJPGS := $(patsubst ${CAT_DIR}/%.jpg, ${GEN_DIR}/%.jpg, $(JPGS))

TOJPGS := $(patsubst ${CAT_DIR}/%.jpg, ${TO_DIR}/%.jpg, $(JPGS))
TSJPGS := $(patsubst ${CAT_DIR}/%.jpg, ${TS_DIR}/%.jpg, $(JPGS))

SIZE = 50

.PHONY: directories

none:

cleantrain:
	rm ../TRAIN

gen: directories ${GEN_DIR} $(GJPGS)

gentrain: directories $(TOJPGS) $(TSJPGS)

gendata: gentrain ${DATATO} ${DATATS}

$(DATATO): $(TOJPGS)
	./prepare_data.py $(TO_DIR) $@

$(DATATS): $(TSJPGS)
	./prepare_data.py $(TS_DIR) $@

$(GEN_DIR)/%.jpg: ${CAT_DIR}/%.jpg ${CAT_DIR}/%.jpg.cat
	./draw_simple.py $< $(word 2,$^) $@

directories:
	mkdir -p ${TS_DIR}
	mkdir -p ${TO_DIR}
	mkdir -p ${GEN_DIR}

$(TS_DIR)/%.jpg: ${CAT_DIR}/%.jpg ${CAT_DIR}/%.jpg.cat
	./resize.py ${SIZE} $< $(word 2, $^) $@

$(TO_DIR)/%.jpg: ${GEN_DIR}/%.jpg ${CAT_DIR}/%.jpg.cat
	./resize.py ${SIZE} $< $(word 2, $^) $@

