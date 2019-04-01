# Got from https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/

CC := g++
SRCDIR := src
BUILDDIR := build
TARGETDIR := bin

TARGET := $(TARGETDIR)/lm
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -Wall -std=c++11
INC := -I include

# LIB := -pthread -lmongoclient -L lib -lboost_thread-mt -lboost_filesystem-mt -lboost_system-mt
LIB := -pthread


$(TARGET): $(OBJECTS)
	@echo " Linking...";
	@mkdir -p $(TARGETDIR)
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning...";
	$(RM) -r $(BUILDDIR) $(TARGETDIR)


.PHONY: clean
