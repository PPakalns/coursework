default: none

CAT_DIR := ../CAT
GEN_DIR := ../GEN

JPGS := $(wildcard ${CAT_DIR}/*.jpg)
GJPGS := $(patsubst ${CAT_DIR}/%.jpg, ${GEN_DIR}/%.jpg, $(JPGS))

none:

gen: $(GJPGS)

${GEN_DIR}:
	mkdir ${GEN_DIR}

$(GEN_DIR)/%.jpg: ${CAT_DIR}/%.jpg ${CAT_DIR}/%.jpg.cat ${GEN_DIR}
	./draw_simple.py $< $(word 2,$^) $@


