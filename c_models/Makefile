APPS = ome_vertex.mk ihcan_vertex.mk drnl_vertex.mk an_vertex.mk

all: $(APPS)
	for f in $(APPS); do $(MAKE) -f $$f || exit $$?; done

%.aplx: %.mk
	"$(MAKE)" -f $*

clean: $(DIRS)
	for f in $(APPS); do $(MAKE) -f $$f clean || exit $$?; done