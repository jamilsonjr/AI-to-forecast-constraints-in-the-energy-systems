# p_load_profile
# q_load_profile
# p_gen_profile   
# q_gen_profile
# Plot load and gen profiles in (2, 1) subplots.
fig, ax = plt.subplots(2, 1, figsize=(30, 20))
ax[0].plot(timestamps_index, p_load_profile.sum(axis=1))
ax[0].set_title('P Load profile (kW)')
ax[0].grid()
ax[1].plot(timestamps_index, p_gen_profile.sum(axis=1))
ax[1].set_title('P Generator profile (kW)')
ax[1].grid()
plt.show()
# Same for reactive power
fig, ax = plt.subplots(2, 1, figsize=(30, 20))
ax[0].plot(timestamps_index, q_load_profile.sum(axis=1))
ax[0].set_title('Q Load profile (kVAr)')
ax[0].grid()
ax[1].plot(timestamps_index, q_gen_profile.sum(axis=1))
ax[1].set_title('Q Generator profile (kVAr)')
ax[1].grid()
plt.show()

####
# Plot Active and Reactive Power for the day of 15th of August 2020.
# Time stamps index for the day of 15th of August 2020.
day_of_15_aug_2020 = timestamps_index[(timestamps_index.day == 15)
                                      & (timestamps_index.month == 8)
                                      & (timestamps_index.year == 2020)]