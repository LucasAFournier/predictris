 

/**
 * Double buffered JPanel with
 * volatile images for performance
 * and quality improvement
 * 
 * @author Jean-Charles Quinton
 * @version 11/03/2008
 * creation 23/03/2007
 */

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.awt.image.*;

abstract public class JPanelDB extends JPanel {

    /** Double buffer on VRAM handling */
    VolatileImage backBuffer = null;

    /** Constructor */
    public JPanelDB() {
        super();
        // Interactions
//         MouseAdapter adapter = new MouseInteract();
//         addMouseMotionListener(adapter);
//         addMouseListener(adapter);
//         addMouseWheelListener(adapter);
//         addKeyListener(new KeyInteract());
    }

    /** Key and mouse listeners and interactions */
    class MouseInteract extends MouseAdapter {
        public void mousePressed(MouseEvent e) {
            interact(e);
        }
    }

    class KeyInteract extends KeyAdapter {            
        public void keyPressed(KeyEvent e) {
            interact(e);
        }
    }
    
    /** Interactions with the interface */
    abstract public void interact(EventObject event);

    /** Creation of a back buffer if not existing */
    void createBackBuffer() {
        if (backBuffer != null) {
            backBuffer.flush();
            backBuffer = null;
        }
        backBuffer = createVolatileImage(getWidth(), getHeight());
    }

    public void paint(Graphics g) {
        if (backBuffer == null || backBuffer.getWidth()!=getWidth() || backBuffer.getHeight()!=getHeight()) {
            createBackBuffer();
        }
        do {
            int valCode = backBuffer.validate(getGraphicsConfiguration());
            if (valCode == VolatileImage.IMAGE_INCOMPATIBLE) {
                createBackBuffer();
            }
            Graphics gBB = backBuffer.getGraphics();
            Graphics2D g2d = (Graphics2D)gBB;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
            render(g2d);
            g.drawImage(backBuffer, 0, 0, this);
        } while (backBuffer.contentsLost());
    }

    /** True repaint function where commands
     *  to draw elements are to be put
     *  (this function should not be called directly)
     *  @param  g   graphics environment given by the paint function */
    abstract public void render(Graphics2D g);
}