import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from midiutil import MIDIFile


def detect_notes_with_pitch(audio_file, sr=None, hop_length=512, threshold=0.5, min_note_distance_sec=0.1, max_note_duration_sec=5.0):
    
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr)
    
    # Compute spectrogram using a short time DFT
    S = np.abs(librosa.stft(y, n_fft=8096, hop_length=hop_length))
    
    # Compute spectral flux for onset detection
    diff = np.diff(S, axis=1)
    diff = np.maximum(0, diff)
    flux = np.sum(diff, axis=0)
    flux = flux / np.max(flux)
    flux = np.concatenate(([0], flux))
    
    # Smooth the flux
    window_size = 5
    flux_smoothed = np.convolve(flux, np.ones(window_size)/window_size, mode='same')

    # Calculate adaptive threshold
    window_size = 30
    flux_mean = np.convolve(flux_smoothed, np.ones(window_size)/window_size, mode='same')
    flux_std = np.zeros_like(flux_mean)

    for i in range(len(flux_std)):
        start = max(0, i - window_size//2)
        end = min(len(flux), i + window_size//2 + 1)
        flux_std[i] = np.std(flux_smoothed[start:end])

    adaptive_threshold = np.maximum(flux_mean + 0.7 * flux_std, threshold)

    # Find peaks above threshold
    peaks = []
    for i in range(1, len(flux_smoothed)-1):
        if flux_smoothed[i] > adaptive_threshold[i] and flux_smoothed[i] > flux_smoothed[i-1] and flux_smoothed[i] > flux_smoothed[i+1]:
            peaks.append(i)
    
    # Convert frame indices to time
    onset_frames = np.array(peaks)
    min_frames_between_notes = int(min_note_distance_sec * sr / hop_length)
    
    # Filter out onsets that are too close (should not be needed after implementing the smoothing but keeping for safety)
    if len(onset_frames) > 0:
        filtered_onsets = [onset_frames[0]]
        last_onset = onset_frames[0]
        
        for onset in onset_frames[1:]:
            if onset - last_onset >= min_frames_between_notes:
                filtered_onsets.append(onset)
                last_onset = onset
        
        onset_frames = np.array(filtered_onsets)
    
    # Convert to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    # List to store detected notes
    notes = []
    
    # For each detected onset
    for i, (frame, time) in enumerate(zip(onset_frames, onset_times)):
        # Determine note duration
        if i < len(onset_frames) - 1:
            # Duration until next onset or max duration
            max_frames = int(max_note_duration_sec * sr / hop_length)
            next_onset = onset_frames[i+1]
            note_end_frame = min(frame + max_frames, next_onset)
        else:
            # For last note, use max duration or extend to end
            max_frames = int(max_note_duration_sec * sr / hop_length)
            note_end_frame = min(frame + max_frames, S.shape[1])
        
        # Extract the spectrogram segment for this note
        note_spec = S[:, frame:note_end_frame]
        
        # Find pitch by locating the bin with maximum energy
        if note_spec.size > 0:
            # Sum across time to get frequency profile
            freq_profile = np.sum(note_spec, axis=1)
            
            # Find the frequency bin with maximum energy
            max_bin = np.argmax(freq_profile)
            
            # Convert bin to frequency
            freq = librosa.fft_frequencies(sr=sr, n_fft=2*(S.shape[0]-1))[max_bin]
            
            # Convert frequency to MIDI note
            if freq > 0:
                midi_note = int(round(69 + 12 * np.log2(freq / 440.0)))
                
                # Ensure it's in the valid MIDI range (0-127)
                midi_note = max(0, min(127, midi_note))
            else:
                midi_note = 0
                
        # Calculate velocity based on spectral energy in the onset window
        start_frame = frame
        end_frame = min(frame + 3, S.shape[1])  # Look at first ~3 frames after onset

        if start_frame < S.shape[1]:
            # Get the spectral content for this segment
            onset_spectrum = S[:, start_frame:end_frame]
            
            if onset_spectrum.size > 0:
                # Calculate velocity based on peak spectral energy
                peak_energy = np.max(onset_spectrum)
                # Normalize to MIDI velocity range (1-127)
                velocity = int(127 * peak_energy / np.max(S))
                velocity = max(1, min(127, velocity))
            else:
                velocity = 64
        else:
            velocity = 64


        # If no frequency found, set to 0                
        # Store this note
        notes.append({
            'onset_time': time,
            'midi_note': midi_note,
            'frequency': freq,
            'velocity': velocity,
            'duration': librosa.frames_to_time(note_end_frame - frame, sr=sr, hop_length=hop_length)
        })
    
    return notes, flux_smoothed, adaptive_threshold

def save_notes_to_midi(notes, output_file, tempo=120):

    # Create MIDI file with 1 track
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    
    # Add tempo
    midi.addTempo(track, time, tempo)
    
    # Add notes
    for note in notes:
        # Convert time in seconds to beats
        start_time = note['onset_time'] * (tempo / 60.0)  # Convert seconds to beats
        duration = note['duration'] * (tempo / 60.0)      # Convert seconds to beats
        
        # Add note to MIDI file
        midi.addNote(
            track=track,
            channel=channel,
            pitch=note['midi_note'],
            time=start_time,
            duration=duration,
            volume=note['velocity']
        )
    
    # Save MIDI file
    with open(output_file, "wb") as f:
        midi.writeFile(f)

def visualize_detected_notes(y, sr, notes, flux=None, threshold=None):

    plt.figure(figsize=(16, 12))
   
    # Plot waveform
    plt.subplot(5, 1, 1)
    librosa.display.waveshow(y, sr=sr)
   
    # Add onsets if any notes were detected
    if notes:  # Check if notes list is not empty
        onset_times = [note['onset_time'] for note in notes]
        plt.vlines(onset_times, -1, 1, color='r', linestyle='--', label='Onsets')
        plt.title('Waveform with Detected Onsets')
        plt.legend()
    else:
        plt.title('Waveform (No Onsets Detected)')
   
    # Plot spectrogram with detected notes
    plt.subplot(5, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(img, format='%+2.0f dB')
   
    # Add detected notes to spectrogram if any
    if notes:
        onset_times = [note['onset_time'] for note in notes]
        freqs = [note['frequency'] for note in notes]
        plt.scatter(onset_times, freqs, color='cyan', s=30, marker='x')
       
        # Optionally draw horizontal lines at detected frequencies
        for time, freq in zip(onset_times, freqs):
            plt.axvline(x=time, color='w', linestyle='--', alpha=0.5)
   
    plt.title('Spectrogram with Detected Notes')
   
    # Plot flux and threshold
    if flux is not None and threshold is not None:
        plt.subplot(5, 1, 3)
        times = librosa.times_like(flux, sr=sr, hop_length=512)
        plt.plot(times, flux, label='Spectral Flux')
        
        # Handle both single value threshold and array threshold
        if np.isscalar(threshold):
            # For a single threshold value, plot a horizontal line
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        else:
            # For an adaptive threshold array, plot the curve
            plt.plot(times, threshold, 'r--', label='Adaptive Threshold')
            
        plt.legend()
        plt.title('Spectral Flux and Threshold')
   
    # Plot pitches and velocities only if notes were detected
    if notes:
        # Plot pitches
        plt.subplot(5, 1, 4)
        midi_notes = [note['midi_note'] for note in notes]
        onset_times = [note['onset_time'] for note in notes]
        plt.scatter(onset_times, midi_notes)
        plt.ylabel('MIDI Note')
        plt.title('Detected Note Pitches')
       
        # Plot velocities
        plt.subplot(5, 1, 5)
        velocities = [note['velocity'] for note in notes]
        plt.scatter(onset_times, velocities)
        plt.xlabel('Time (s)')
        plt.ylabel('MIDI Velocity')
        plt.title('Note Velocities')
    else:
        # Show information message if no notes were detected
        plt.subplot(5, 1, 4)
        plt.text(0.5, 0.5, 'No notes detected - try adjusting threshold parameters',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.axis('off')
       
        plt.subplot(5, 1, 5)
        plt.axis('off')
   
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace with your harp recording
    audio_file = "C:/4CA10/Angel of music piano.wav"
    
    # Detect notes
    notes, flux_smoothed, threshold = detect_notes_with_pitch(
        audio_file, 
        threshold=0.04,  # Try different values if needed
        min_note_distance_sec=0.1
    )
    
    # Load audio for visualization
    y, sr = librosa.load(audio_file)
    
    # Print detected notes
    print(f"Detected {len(notes)} notes:")
    for i, note in enumerate(notes):
        note_name = librosa.midi_to_note(note['midi_note'])
        print(f"Note {i+1}: {note_name} at {note['onset_time']:.2f}s, velocity {note['velocity']}, duration {note['duration']:.2f}s")
    
    # Save to MIDI file
    output_midi = "output.mid"
    save_notes_to_midi(notes, output_midi, tempo=120)
    print(f"MIDI file saved as {output_midi}")


    # Visualize results
    visualize_detected_notes(y, sr, notes, flux_smoothed, threshold)