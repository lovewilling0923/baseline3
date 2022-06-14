#!/bin/bash
ps -ef | grep python | awk '{print $2}' | xargs kill -9
ps -ef | grep ray | awk '{print $2}' | xargs kill -9
ps -ef | grep fps.x86_64 | awk '{print $2}' | xargs kill -9
ps -ef | grep htop | awk '{print $2}' | xargs kill -9
