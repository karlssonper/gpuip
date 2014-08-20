#!/bin/bash
mkdir -p ~/tmp/
git clone git@github.com:karlssonper/gpuip -b gh-pages ~/tmp/gpuip-web
cur=$(pwd)
echo $cur
cd ~/tmp/gpuip-web
git pull origin
git rm -r api/*
mkdir api
cp -r $cur/html/* api/
git add api/*
git commit -m "api update"
git push
cd $cur
rm -rf ~/tmp/gpuip-web