import sys
if __name__ == '__main__':
	s = sys.argv[1]
	o = sys.argv[2]
	max_count = 100000
	email_idx = -1
	tag_idx = -1
	count = 0
	tmp_count = 0
	data = []
	with open(s, "r") as f:
		for line in f:
			line = line.strip()
			l = line.split('\x07')
			if email_idx == -1 and tag_idx == -1:
				email_idx = l.index("EmailAliases")
				tag_idx = l.index("is_swm_bad")
			else:
				if len(l) > max(email_idx, tag_idx):
					emails = l[email_idx]
					tag = l[tag_idx]
					if emails:
						emails = emails.split()
						tmp_count += len(emails)
						count += len(emails)
						for email in emails:
							data.append(email+"\x07"+str(tag)+"\n")
						if tmp_count > max_count:
							print("Deal %d emails"%(count))
							with open(o,"a") as out:
								out.writelines(data)
							data = []
							tmp_count = 0
		else:
			print("Deal %d emails"%(count))
			with open(o,"a") as out:
				out.writelines(data)
	print("totally, there are %d emails"%(count))
