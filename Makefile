CC = g++
CFLAGS = -std=c++14 -pedantic  `pkg-config opencv --cflags --libs`
BIN = exo
# g++ -o test_1 exo2.cc `pkg-config opencv --cflags --libs`

SRC = test.cpp

OBJ = $(SRC:.c=.o)


.PHONY: clean

all: $(OBJ)
	$(CC) $(CFLAGS) -o $(BIN) $^

debug: $(OBJ)
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) -o $(BIN) $^

clean:
	$(RM) src/*o $(BIN)
