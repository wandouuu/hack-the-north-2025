# File: qnx_stream_receiver.py - Modified for downscaled frames with headers
import socket
import numpy as np
import cv2
import struct
import time

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 12345

# --- IMPORTANT: Match these to your C client settings ---
DOWNSCALE_WIDTH =  640  # Must match C code
DOWNSCALE_HEIGHT = 480  # Must match C code
BYTES_PER_PIXEL = 3     # BGR format (not BGRX)
FRAME_SIZE_BYTES = DOWNSCALE_WIDTH * DOWNSCALE_HEIGHT * BYTES_PER_PIXEL

# Frame header constants
MAGIC_NUMBER = 0xDEADBEEF


'''
Purpose: receive an exact amount of frame data
Arguments: 
'''
def receive_exact(sock, num_bytes):
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def frame_server():
    """
    Generator that yields (ret, image) for each received frame.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        
        print(f"[Python Server] Listening on {HOST}:{PORT}...")
        print(f"[Python Server] Expecting {DOWNSCALE_WIDTH}x{DOWNSCALE_HEIGHT} BGR frames")
        
        conn, addr = s.accept()
        
        with conn:
            print(f"[Python Server] Connected to {addr}")
            
            while True:
                try:
                    # --- Receive frame header ---
                    header_data = receive_exact(conn, 8)  # 4 bytes magic + 4 bytes size
                    if not header_data:
                        print("\n[Python Server] Connection closed.")
                        break
                    
                    magic, frame_size = struct.unpack('<II', header_data)
                    
                    # # Verify magic number for synchronization
                    if magic != MAGIC_NUMBER:
                        print(f"[Warning] Invalid magic number: 0x{magic:08X}, expected 0x{MAGIC_NUMBER:08X}")
                        continue
                    
                    # Verify frame size
                    if frame_size != FRAME_SIZE_BYTES:
                        print(f"[Warning] Unexpected frame size: {frame_size}, expected {FRAME_SIZE_BYTES}")
                        continue
                    
                    # --- Receive frame data ---
                    frame_data = receive_exact(conn, frame_size)
                    if not frame_data:
                        print("\n[Python Server] Connection closed during frame receive.")
                        break
                    
                    # --- Process and display frame ---
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    image = frame_array.reshape((DOWNSCALE_HEIGHT, DOWNSCALE_WIDTH, BYTES_PER_PIXEL))
                    
                    yield True, image

                except ConnectionResetError:
                    print("\n[Python Server] Connection reset by client.")
                    yield False, np.zeros((DOWNSCALE_HEIGHT, DOWNSCALE_WIDTH, BYTES_PER_PIXEL), dtype=np.uint8)
                    break
                except Exception as e:
                    print(f"\n[Python Server] Error: {e}")
                    yield False, np.zeros((DOWNSCALE_HEIGHT, DOWNSCALE_WIDTH, BYTES_PER_PIXEL), dtype=np.uint8)
                    break

    cv2.destroyAllWindows()
    print("[Python Server] Server shut down.")

if __name__ == "__main__":
    print("QNX Camera Stream Receiver")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("-" * 40)
    frame_gen = frame_server()
    frame_count = 0
    last_fps_time = time.time()
    for ret, image in frame_gen:
        if ret:
            # Convert BGR to RGB for proper display (OpenCV uses BGR by default)
            # Actually, since we're sending BGR and OpenCV expects BGR, no conversion needed
            
            # Scale up for better visibility (optional)
            display_image = cv2.resize(image, (DOWNSCALE_WIDTH * 2, DOWNSCALE_HEIGHT * 2), 
                                     interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow('QNX Camera Stream', display_image)
            
            # FPS calculation and display
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:  # Update every second
                fps = frame_count / (current_time - last_fps_time)
                print(f"[FPS: {fps:.1f}] Frames received: {frame_count}")
                frame_count = 0
                last_fps_time = current_time
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Python Server] 'q' key pressed, shutting down.")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"frame_{timestamp}.png"
                cv2.imwrite(filename, display_image)
                print(f"[Python Server] Frame saved as {filename}")

        else:
            print("[Python Server] Failed to receive frame.")