import winsound

def play_alarm():
    duration = 500  # ms
    freq = 1000     # Hz
    winsound.Beep(freq, duration)