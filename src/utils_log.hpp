#ifndef UTILS_LOG_HPP
#define UTILS_LOG_HPP

#include <string>
#include <cstring>
#include <iostream>

#include "type.h"

class UtilsLog
{
private:
    st::LogLevel log_level_;
    time_t time_; 

    const std::string show_realtime(time_t t) {
        static char buf[32];
        char *p = NULL;

        time(&t);
        strcpy(buf, ctime(&t));
        p = strchr(buf, '\n');
        *p = '\0';
        
        std::string out(buf);

        return out;
    }

public:
    UtilsLog() {};
    ~UtilsLog() {};

    void open(const st::LogLevel &log_level) {
        log_level_ = log_level;
    }

    void Info(const char *prefix, const std::string str) {
        if (log_level_ == st::INFO || log_level_ == st::DEBUG) {
            std::cout << "\033[33m" << "<INFO>" << "[" << show_realtime(time_) \
                      << "]"<< " " << prefix << "::" << str << "\033[0m" << std::endl; 
        }
    }
    void Debug(const char *prefix, const std::string str) {
        if (log_level_ == st::DEBUG) {
            std::cout << "\033[32m" << "<DEBUG>" << "[" << show_realtime(time_) \
                      << "]"<< " " << prefix << "::" << str << "\033[0m" << std::endl;
        }
    }
    void Error(const char *prefix, const std::string str) {
        std::cout << "\033[31m" << "<ERROR>" << "[" << show_realtime(time_) \
                    << "]"<< " " << prefix << "::" << str << "\033[0m" << std::endl;
    }
};

#endif // UTILS_LOG_HPP