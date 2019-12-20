from source.signal_filtering import Filter

r3_filter = Filter(3)
r3_filter.filter()

r5_filter = Filter(5)
r5_filter.filter()

r3_filter.visualize_filtering()
r5_filter.visualize_filtering()
