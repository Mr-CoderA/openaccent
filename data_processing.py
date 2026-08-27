from files import Files

verify_files = Files()
status_clean = verify_files.clean()['status']
verify = verify_files.verify()
format_check = verify_files.format_check()
sample_rate = format_check.get('sample_rate')

if verify.get('status') == 200 and sample_rate == 22050 and status_clean == 200:
    print("VERIFIED", verify)
    print('----- DATA SPLIT -----')
    data_files = verify_files.get_train_eval()
    print(data_files)
    print("----- WAV to MEL -----")
    verify_files.wav_to_mel(sample_rate)
else:
    print("NOT VERIFIED", verify, sample_rate, status_clean)