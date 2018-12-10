#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "log5cxx.h"

#define COMMAND_HELPER(x) { ""#x"", cmd_##x, cmd_##x##_usage, 1}
#define COMMAND_DECLEAR(x) extern int cmd_##x(int argc, char **argv); \
                           extern void cmd_##x##_usage();

typedef struct cmd_struct {
    const char *cmd;
    int (*fn)(int, char **);
    void (*ufn)();
    int option;
};


COMMAND_DECLEAR(fingerprint)
COMMAND_DECLEAR(shotdetect)

static struct cmd_struct commands[] = {
    COMMAND_HELPER(fingerprint),
    COMMAND_HELPER(shotdetect),
};
static const int commands_len = 2;

void copyright();
void help();
int main(int argc, char** argv){
    LOG5CXX_INIT(argv[0]);

    if (argc < 2){
        help();
        return 0;
        //LOG5CXX_FATAL("fatal: argument error.", 1);
    }

    int i;
    for (i=0; i<commands_len; i++) {
        cmd_struct *cmd = commands+i;

        if (strcmp(cmd->cmd, argv[1]) == 0)
            return cmd->fn(argc, argv);
    }

    help();
    return 0;
}

void copyright(){
    printf("\
immatch (Built by gcc-g++-builds project) 0.2.0 (alias xxx).\n\
Copyright (C) 2014 cdvcloud.com, Inc.\n\
This is nonfree software; connect to cdvcloud for using conditions.\n\
Author: Charlie Niel.\n\n");
}

void help(){
    printf("\
Usage: immatch <module_name> [options...]\n\
Options:\n\
 -h/--help          This help text\n\
 -v/--version       Show version number and quit\n\n\
");
    for (int i=0; i<commands_len; i++) {
        cmd_struct *cmd = commands+i;
        cmd->ufn();
    }
}