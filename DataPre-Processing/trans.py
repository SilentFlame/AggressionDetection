#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import csv
from indictrans import Transliterator
trn = Transliterator(source='hin', target='eng', build_lookup=True)

# print eng


with open('dataSet.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		hin = row[1].decode('utf-8').replace('\n', " ")
		eng = trn.transform(hin)
		print row[0], eng.encode('unicode-escape'), row[2]
