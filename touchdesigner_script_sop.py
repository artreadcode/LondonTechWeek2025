def onCook(scriptOp):
    # Clear previous geometry
    scriptOp.clear()
    
    # Debug: Print to see if the function is being called
    print("Script SOP onCook called")
    
    try:
        # Get the CHOP with face positions
        face_chop = op('facePos')
        
        if face_chop is None:
            print("ERROR: Could not find CHOP 'facePos'")
            return
        
        # Debug: Print available channels
        print("Available channels:", [chan.name for chan in face_chop.chans()])
        
        # Collect all valid face coordinates
        face_coords = []
        for i in range(10):
            try:
                x_chan = face_chop['face{}_x'.format(i)]
                y_chan = face_chop['face{}_y'.format(i)]
                
                if x_chan is None or y_chan is None:
                    continue
                    
                x = x_chan.eval()  # Use eval() instead of [0]
                y = y_chan.eval()
                
                # Convert normalized coordinates (0-1) to centered coordinates (-1 to 1)
                x_centered = (x * 2.0) - 1.0
                y_centered = (y * 2.0) - 1.0
                
                # Skip if coordinates are at origin (likely no face detected)
                if abs(x) > 0.001 and abs(y) > 0.001:  # Small threshold instead of exact zero
                    face_coords.append((x_centered, y_centered, 0))
                    print(f"Face {i}: ({x_centered:.3f}, {y_centered:.3f})")
                    
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue
        
        print(f"Found {len(face_coords)} valid faces")
        
        if len(face_coords) < 2:
            print("Need at least 2 faces to create connections")
            return
        
        import random
        
        # Create some connections between faces (random pairs)
        pairs = []
        for i in range(len(face_coords)):
            for j in range(i+1, len(face_coords)):
                pairs.append((i, j))
        
        # Limit number of connections to avoid visual clutter
        max_connections = min(5, len(pairs))
        selected_pairs = random.sample(pairs, max_connections) if pairs else []
        
        print(f"Creating {len(selected_pairs)} connections")
        
        # For each pair, draw a bezier curve
        for pair_idx, (i, j) in enumerate(selected_pairs):
            try:
                p0_coords = face_coords[i]
                p3_coords = face_coords[j]
                
                # Create control points for Bezier handles
                mid_x = (p0_coords[0] + p3_coords[0]) / 2
                mid_y = (p0_coords[1] + p3_coords[1]) / 2
                mid_z = 0.2  # Slight elevation for visual interest
                
                # Control points that create a gentle curve
                p1_x = p0_coords[0] * 0.7 + mid_x * 0.3
                p1_y = p0_coords[1] * 0.7 + mid_y * 0.3
                p1_z = 0.1
                
                p2_x = p3_coords[0] * 0.7 + mid_x * 0.3
                p2_y = p3_coords[1] * 0.7 + mid_y * 0.3
                p2_z = 0.1
                
                # Create the Bezier curve
                curve = scriptOp.appendBezier()
                curve.insertPoint(0)  # Start point
                curve.insertPoint(1)  # Control point 1
                curve.insertPoint(2)  # Control point 2
                curve.insertPoint(3)  # End point
                
                # Set the points
                curve.point(0).x = p0_coords[0]
                curve.point(0).y = p0_coords[1]
                curve.point(0).z = p0_coords[2]
                
                curve.point(1).x = p1_x
                curve.point(1).y = p1_y
                curve.point(1).z = p1_z
                
                curve.point(2).x = p2_x
                curve.point(2).y = p2_y
                curve.point(2).z = p2_z
                
                curve.point(3).x = p3_coords[0]
                curve.point(3).y = p3_coords[1]
                curve.point(3).z = p3_coords[2]
                
                print(f"Created curve {pair_idx + 1}")
                
            except Exception as e:
                print(f"Error creating curve for pair {i},{j}: {e}")
                continue
                
    except Exception as e:
        print(f"General error in onCook: {e}")
        import traceback
        traceback.print_exc()
