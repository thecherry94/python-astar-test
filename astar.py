import pygame
import math
from queue import PriorityQueue # For A*
from collections import deque # For Flood Fill BFS

# --- Pygame Setup ---
GRID_WIDTH = 750
INFO_PANEL_WIDTH = 350
TOTAL_WIDTH = GRID_WIDTH + INFO_PANEL_WIDTH
WINDOW_HEIGHT = 750

WIN = pygame.display.set_mode((TOTAL_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("A* Pathfinding with Terrain & Flood Fill")
pygame.font.init()

# --- Colors ---
INFO_PANEL_BG = (230, 230, 230)
TEXT_COLOR = (10, 10, 10)
HIGHLIGHT_BORDER = (0, 100, 255)
BUTTON_BORDER_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (200, 200, 255)
BUTTON_SELECTED_BG_COLOR = (200, 200, 250) # Light blue for selected

RED_CLOSED = (255, 150, 150)
GREEN_OPEN = (150, 255, 150)
PURPLE_PATH = (128, 0, 128)
ORANGE_START = (255, 165, 0)
TURQUOISE_END = (64, 224, 208)
BLACK_OBSTACLE = (50, 50, 50)

# --- Terrain Definitions ---
TERRAIN_TYPES = {
    "GRASS":    ("Grass", (34, 139, 34), 1.0),
    "ROAD":     ("Road", (160, 160, 160), 0.5),
    "DIRT":     ("Dirt", (139, 69, 19), 2.0),
    "WATER":    ("Water", (30, 144, 255), 5.0),
    "OBSTACLE": ("Obstacle", BLACK_OBSTACLE, float('inf')),
}
DEFAULT_TERRAIN = "GRASS"

# --- Fonts ---
NODE_F_COST_FONT = pygame.font.SysFont('arial', 13)
INFO_HEADER_FONT = pygame.font.SysFont('arial', 20, bold=True)
INFO_FONT = pygame.font.SysFont('arial', 16)
LEGEND_FONT = pygame.font.SysFont('arial', 14)
STATUS_FONT = pygame.font.SysFont('arial', 18, bold=True)
BRUSH_FONT = pygame.font.SysFont('arial', 15) # For brush buttons
MODE_FONT = pygame.font.SysFont('arial', 15) # For paint mode buttons

# --- Brush Types (user selection) ---
BRUSH_START = "START"
BRUSH_END = "END"
# Terrain keys ("GRASS", "ROAD", etc.) also act as brush types.

# --- Paint Modes ---
PAINT_MODE_SINGLE = "SINGLE_TILE"
PAINT_MODE_FLOOD = "FLOOD_FILL"

class Node:
    def __init__(self, row, col, cell_width, total_rows):
        self.row = row
        self.col = col
        self.draw_x = col * cell_width
        self.draw_y = row * cell_width
        self.cell_width = cell_width
        self.total_rows = total_rows

        self.terrain_type = DEFAULT_TERRAIN
        self.base_color = TERRAIN_TYPES[self.terrain_type][1]
        self.movement_cost = TERRAIN_TYPES[self.terrain_type][2]
        
        self.current_display_color = self.base_color
        self.is_special_node_type = None # Can be BRUSH_START or BRUSH_END

        self.neighbors = []
        self.parent = None
        self.g_cost = float("inf")
        self.h_cost = float("inf")
        self.f_cost = float("inf")

        self.is_in_open_set = False
        self.is_in_closed_set = False
        self.is_on_path = False

    def get_pos(self):
        return self.row, self.col

    def _update_display_color(self):
        if self.is_special_node_type == BRUSH_START: self.current_display_color = ORANGE_START
        elif self.is_special_node_type == BRUSH_END: self.current_display_color = TURQUOISE_END
        elif self.is_on_path: self.current_display_color = PURPLE_PATH
        elif self.is_in_open_set: self.current_display_color = GREEN_OPEN
        elif self.is_in_closed_set: self.current_display_color = RED_CLOSED
        else: self.current_display_color = self.base_color

    def set_terrain(self, terrain_key):
        if terrain_key in TERRAIN_TYPES:
            self.terrain_type = terrain_key
            self.base_color = TERRAIN_TYPES[terrain_key][1]
            self.movement_cost = TERRAIN_TYPES[terrain_key][2]
            
            # If this node was Start or End, it loses that status when terrain is painted over
            # The main loop handles updating start_node/end_node references if this happens
            if self.is_special_node_type:
                self.is_special_node_type = None 
            
            self._reset_astar_states()
            self._update_display_color() # Will set to base_color if not special

    def _reset_astar_states(self):
        self.g_cost = float("inf")
        self.h_cost = float("inf")
        self.f_cost = float("inf")
        self.parent = None
        self.is_in_open_set = False
        self.is_in_closed_set = False
        self.is_on_path = False

    def reset_to_default_terrain(self):
        if self.is_special_node_type: # Handle if Start/End is reset
            self.is_special_node_type = None
        self.set_terrain(DEFAULT_TERRAIN) # This will clear A* states and update color

    def make_special_node(self, node_type): # BRUSH_START or BRUSH_END
        self.is_special_node_type = node_type
        if node_type == BRUSH_START: self.g_cost = 0 # Start node has g_cost 0
        else: self._reset_astar_states() # End node starts with normal A* values
        
        # Keep underlying terrain, but visuals/pathfinding logic uses special status
        self.is_in_open_set = False
        self.is_in_closed_set = False
        self.is_on_path = False
        self._update_display_color()

    def make_obstacle(self):
        self.set_terrain("OBSTACLE")

    def set_open(self):
        self.is_in_open_set = True; self.is_in_closed_set = False; self.is_on_path = False
        self._update_display_color()

    def set_closed(self):
        self.is_in_open_set = False; self.is_in_closed_set = True; self.is_on_path = False
        self._update_display_color()

    def set_path(self):
        self.is_in_open_set = False; self.is_in_closed_set = True; self.is_on_path = True
        self._update_display_color()

    def draw(self, win, hovered=False):
        pygame.draw.rect(win, self.current_display_color, (self.draw_x, self.draw_y, self.cell_width, self.cell_width))
        if hovered:
             pygame.draw.rect(win, HIGHLIGHT_BORDER, (self.draw_x, self.draw_y, self.cell_width, self.cell_width), 2)

        if self.is_in_open_set and self.f_cost != float('inf') and not self.is_special_node_type:
            f_text_surface = NODE_F_COST_FONT.render(f"{self.f_cost:.1f}", True, BLACK_OBSTACLE)
            text_rect = f_text_surface.get_rect(center=(self.draw_x + self.cell_width // 2, self.draw_y + self.cell_width // 2))
            win.blit(f_text_surface, text_rect)

    def update_neighbors(self, grid):
        self.neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = self.row + dr, self.col + dc
            if 0 <= nr < self.total_rows and 0 <= nc < self.total_rows:
                neighbor_node = grid[nr][nc]
                if neighbor_node.movement_cost != float('inf'):
                    self.neighbors.append(neighbor_node)

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def h(p1, p2):
    r1, c1 = p1; r2, c2 = p2
    return abs(r1 - r2) + abs(c1 - c2)

def algorithm(draw_callback, grid, start_node, end_node, set_status_message_callback):
    count = 0
    open_set = PriorityQueue()
    
    start_node.h_cost = h(start_node.get_pos(), end_node.get_pos())
    start_node.f_cost = start_node.g_cost + start_node.h_cost
    open_set.put((start_node.f_cost, count, start_node))
    
    open_set_hash = {start_node}
    if not start_node.is_special_node_type: start_node.set_open()

    set_status_message_callback("Algorithm Running...")

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return False

        current_node = open_set.get()[2]
        open_set_hash.remove(current_node)
        current_node.is_in_open_set = False

        if current_node == end_node:
            set_status_message_callback(f"Path Found! Cost: {current_node.g_cost:.1f}")
            reconstruct_path(current_node, draw_callback, start_node, end_node)
            return True

        for neighbor in current_node.neighbors:
            cost_to_neighbor = neighbor.movement_cost
            temp_g_cost = current_node.g_cost + cost_to_neighbor

            if temp_g_cost < neighbor.g_cost:
                neighbor.parent = current_node
                neighbor.g_cost = temp_g_cost
                neighbor.h_cost = h(neighbor.get_pos(), end_node.get_pos())
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((neighbor.f_cost, count, neighbor))
                    open_set_hash.add(neighbor)
                    if not neighbor.is_special_node_type: neighbor.set_open()
        
        draw_callback()
        pygame.time.delay(20)

        if current_node != start_node: current_node.set_closed()
        elif current_node == start_node: start_node.make_special_node(BRUSH_START) # Keep start visually start

    set_status_message_callback("Path Not Found.")
    return False

def reconstruct_path(current_node, draw_callback, start_node, end_node):
    path_trace_node = current_node
    while path_trace_node.parent:
        if path_trace_node != end_node: path_trace_node.set_path()
        path_trace_node = path_trace_node.parent
        draw_callback()
        pygame.time.delay(30)
    if start_node: start_node.make_special_node(BRUSH_START)
    if end_node: end_node.make_special_node(BRUSH_END)
    draw_callback()

def make_grid(rows, grid_pixel_width):
    grid = []
    gap = grid_pixel_width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            grid[i].append(Node(i, j, gap, rows))
    return grid

def draw_grid_lines(win, rows, grid_pixel_width, window_height_for_lines):
    gap = grid_pixel_width // rows
    for i in range(rows + 1):
        pygame.draw.line(win, (200,200,200), (0, i * gap), (grid_pixel_width, i * gap))
        pygame.draw.line(win, (200,200,200), (i * gap, 0), (i * gap, window_height_for_lines))

# --- Flood Fill Implementation ---
def run_flood_fill(grid, rows, start_r, start_c, fill_terrain_key, current_start_node, current_end_node):
    """
    Performs a flood fill on the grid.
    - grid: The 2D list of Node objects.
    - rows: Number of rows/cols in the grid.
    - start_r, start_c: Coordinates of the clicked node to start the fill.
    - fill_terrain_key: The terrain key (e.g., "ROAD") to fill with.
    - current_start_node, current_end_node: References to the actual start/end nodes.
    Returns True if any changes were made, False otherwise.
    """
    node_to_start_fill = grid[start_r][start_c]
    target_terrain_key = node_to_start_fill.terrain_type # Terrain type to be replaced

    # Basic checks before starting:
    if node_to_start_fill == current_start_node or node_to_start_fill == current_end_node: return False
    if target_terrain_key == fill_terrain_key: return False
    if target_terrain_key == "OBSTACLE" and fill_terrain_key != "OBSTACLE": return False
    
    q = deque([(start_r, start_c)]) # Use deque for efficient queue
    visited = set([(start_r, start_c)])
    changed_something = False

    while q:
        r, c = q.popleft()
        current_node_in_fill = grid[r][c]

        # Ensure we don't process already visited or invalid nodes (should be caught by neighbor check mostly)
        if current_node_in_fill.terrain_type != target_terrain_key: continue # If it changed somehow
        if current_node_in_fill == current_start_node or current_node_in_fill == current_end_node: continue
        if current_node_in_fill.terrain_type == "OBSTACLE" and fill_terrain_key != "OBSTACLE": continue
        
        current_node_in_fill.set_terrain(fill_terrain_key)
        changed_something = True

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-directional neighbors
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < rows and (nr, nc) not in visited:
                neighbor = grid[nr][nc]
                if neighbor.terrain_type == target_terrain_key and \
                   neighbor != current_start_node and neighbor != current_end_node and \
                   not (neighbor.terrain_type == "OBSTACLE" and fill_terrain_key != "OBSTACLE"):
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return changed_something

# --- Info Panel UI Elements ---
brush_buttons = [] 
paint_mode_buttons = [] 

def draw_info_panel(win, hovered_node, show_instructions, status_message, current_brush_type, current_paint_mode):
    global brush_buttons, paint_mode_buttons
    brush_buttons.clear()
    paint_mode_buttons.clear()

    panel_x_start = GRID_WIDTH
    panel_rect = pygame.Rect(panel_x_start, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(win, INFO_PANEL_BG, panel_rect)
    pygame.draw.line(win, (180,180,180), (panel_x_start, 0), (panel_x_start, WINDOW_HEIGHT), 2)

    current_y = 20

    if status_message:
        status_surf = STATUS_FONT.render(status_message, True, TEXT_COLOR)
        status_rect = status_surf.get_rect(centerx=panel_x_start + INFO_PANEL_WIDTH // 2, top=current_y)
        win.blit(status_surf, status_rect)
        current_y += status_surf.get_height() + 15

    # Hovered Node Info
    header_surf = INFO_HEADER_FONT.render("Hovered Node", True, TEXT_COLOR)
    win.blit(header_surf, (panel_x_start + 20, current_y)); current_y += header_surf.get_height() + 5
    if hovered_node:
        terrain_name = TERRAIN_TYPES.get(hovered_node.terrain_type, ["Unknown"])[0]
        state = "Idle"
        if hovered_node.is_special_node_type == BRUSH_START: state = "Start Node"
        elif hovered_node.is_special_node_type == BRUSH_END: state = "End Node"
        elif hovered_node.is_on_path: state = "On Path"
        elif hovered_node.is_in_open_set: state = "In Open Set"
        elif hovered_node.is_in_closed_set: state = "In Closed Set"
        elif hovered_node.terrain_type == "OBSTACLE": state = "Obstacle"
        
        texts = [
            f"Pos: ({hovered_node.row}, {hovered_node.col})", f"Status: {state}",
            f"Terrain: {terrain_name}", f"Move Cost: {hovered_node.movement_cost if hovered_node.movement_cost != float('inf') else 'Inf'}",
            "--- A* Costs ---",
            f"G: {hovered_node.g_cost if hovered_node.g_cost != float('inf') else '-'}",
            f"H: {hovered_node.h_cost if hovered_node.h_cost != float('inf') else '-'}",
            f"F: {hovered_node.f_cost if hovered_node.f_cost != float('inf') else '-'}"
        ]
        for txt in texts:
            surf = INFO_FONT.render(txt, True, TEXT_COLOR if txt != "--- A* Costs ---" else (100,100,100))
            win.blit(surf, (panel_x_start + 30, current_y))
            current_y += surf.get_height() + (3 if txt != "--- A* Costs ---" else 6)
    else:
        none_surf = INFO_FONT.render("Mouse over grid...", True, (150,150,150))
        win.blit(none_surf, (panel_x_start + 30, current_y)); current_y += none_surf.get_height() + 3
    current_y += 15

    # Paint Mode Selection
    mode_header_surf = INFO_HEADER_FONT.render("Paint Mode", True, TEXT_COLOR)
    win.blit(mode_header_surf, (panel_x_start + 20, current_y)); current_y += mode_header_surf.get_height() + 8
    
    mode_button_width = (INFO_PANEL_WIDTH - 60) // 2 # Two buttons side-by-side
    mode_button_height = 28
    modes_to_show = [
        (PAINT_MODE_SINGLE, "Single Tile"),
        (PAINT_MODE_FLOOD, "Flood Fill")
    ]
    current_x_mode = panel_x_start + 20
    for mode_key, text in modes_to_show:
        button_rect = pygame.Rect(current_x_mode, current_y, mode_button_width, mode_button_height)
        is_current_mode = (mode_key == current_paint_mode)
        is_hovered_button = button_rect.collidepoint(pygame.mouse.get_pos())
        
        btn_bg_color = BUTTON_SELECTED_BG_COLOR if is_current_mode else INFO_PANEL_BG
        if not is_current_mode and is_hovered_button: btn_bg_color = BUTTON_HOVER_COLOR

        pygame.draw.rect(win, btn_bg_color, button_rect)
        pygame.draw.rect(win, BUTTON_BORDER_COLOR, button_rect, 1 if not is_current_mode else 2)
        
        text_surf = MODE_FONT.render(text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=button_rect.center)
        win.blit(text_surf, text_rect)
        
        paint_mode_buttons.append({'rect': button_rect, 'mode': mode_key})
        current_x_mode += mode_button_width + 10
    current_y += mode_button_height + 15


    # Brush Selection UI
    brush_header_surf = INFO_HEADER_FONT.render("Paint Brush", True, TEXT_COLOR)
    win.blit(brush_header_surf, (panel_x_start + 20, current_y)); current_y += brush_header_surf.get_height() + 8
    
    button_height = 28; button_padding = 5
    button_x_offset = panel_x_start + 20; button_width = INFO_PANEL_WIDTH - 40

    display_brushes = [
        (BRUSH_START, "Start Node", ORANGE_START), (BRUSH_END, "End Node", TURQUOISE_END),
    ]
    for terrain_key, (name, color, cost) in TERRAIN_TYPES.items():
        text = f"{name}"
        if cost != float('inf'): text += f" (Cost: {cost})"
        else: text = name # Obstacle has no cost shown this way
        display_brushes.append((terrain_key, text, color))

    for brush_key, text, color in display_brushes:
        button_rect = pygame.Rect(button_x_offset, current_y, button_width, button_height)
        is_current_brush = (brush_key == current_brush_type)
        is_hovered_button = button_rect.collidepoint(pygame.mouse.get_pos())
        
        btn_bg_color = BUTTON_SELECTED_BG_COLOR if is_current_brush else INFO_PANEL_BG
        if not is_current_brush and is_hovered_button: btn_bg_color = BUTTON_HOVER_COLOR

        pygame.draw.rect(win, btn_bg_color, button_rect)
        pygame.draw.rect(win, BUTTON_BORDER_COLOR, button_rect, 1 if not is_current_brush else 2)
        
        swatch_rect = pygame.Rect(button_x_offset + 5, current_y + 4, 20, button_height - 8)
        pygame.draw.rect(win, color, swatch_rect); pygame.draw.rect(win, BLACK_OBSTACLE, swatch_rect, 1)

        text_surf = BRUSH_FONT.render(text, True, TEXT_COLOR)
        text_draw_pos_y = current_y + (button_height - text_surf.get_height()) // 2
        win.blit(text_surf, (button_x_offset + 30, text_draw_pos_y))
        
        brush_buttons.append({'rect': button_rect, 'brush': brush_key})
        current_y += button_height + button_padding
    current_y += 10

    # Instructions Toggle
    instr_toggle_surf = INFO_FONT.render(f"Instructions ('I' to {'Hide' if show_instructions else 'Show'}):", True, TEXT_COLOR)
    win.blit(instr_toggle_surf, (panel_x_start + 20, current_y)); current_y += instr_toggle_surf.get_height() + 5
    if show_instructions:
        instructions = [
            "1. Select Paint Mode & Brush.",
            "2. L-Click/Drag (Single Mode) or L-Click (Flood) on grid.",
            "3. R-Click on grid to Erase to Grass.",
            "4. SPACE: Start A* Algorithm.", "5. 'C' Key: Clear grid & reset.",
        ]
        for line in instructions:
            surf = LEGEND_FONT.render(line, True, TEXT_COLOR)
            win.blit(surf, (panel_x_start + 30, current_y)); current_y += surf.get_height() + 2

def draw_all(win, grid, rows, grid_w, hovered_node, show_instr, status_msg, current_brush, current_paint_mode):
    win.fill((255,255,255)) 
    for row_list in grid:
        for node in row_list:
            node.draw(win, hovered=(node == hovered_node))
    draw_grid_lines(win, rows, grid_w, WINDOW_HEIGHT)
    draw_info_panel(win, hovered_node, show_instr, status_msg, current_brush, current_paint_mode)
    pygame.display.update()

def get_clicked_grid_pos(pos, rows, grid_pixel_width):
    mouse_x, mouse_y = pos
    if not (0 <= mouse_x < grid_pixel_width and 0 <= mouse_y < WINDOW_HEIGHT): return None
    gap = grid_pixel_width // rows
    col = mouse_x // gap; row = mouse_y // gap
    return min(max(0, row), rows - 1), min(max(0, col), rows - 1)

def main(win, total_width, grid_pixel_width):
    ROWS = 30
    grid = make_grid(ROWS, grid_pixel_width)

    start_node = None; end_node = None
    algorithm_started = False
    current_hovered_node = None
    show_instructions = True
    status_message = "Select Brush & Paint Mode"
    
    current_brush_type = BRUSH_START
    current_paint_mode = PAINT_MODE_SINGLE
    drawing_on_grid_mode = False # For click-and-drag (single tile paint)

    def set_status(message): nonlocal status_message; status_message = message

    run = True; clock = pygame.time.Clock()
    while run:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        grid_pos_hover = get_clicked_grid_pos(mouse_pos, ROWS, grid_pixel_width)
        current_hovered_node = grid[grid_pos_hover[0]][grid_pos_hover[1]] if grid_pos_hover else None
        
        draw_all(win, grid, ROWS, grid_pixel_width, current_hovered_node, show_instructions, status_message, current_brush_type, current_paint_mode)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: run = False

            if not algorithm_started:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_on_ui = False
                    if mouse_pos[0] >= GRID_WIDTH: # Click in info panel
                        clicked_on_ui = True # Assume UI click if in panel, process below
                        for btn in paint_mode_buttons:
                            if btn['rect'].collidepoint(mouse_pos):
                                current_paint_mode = btn['mode']
                                set_status(f"Mode: {TERRAIN_TYPES.get(current_brush_type, [current_brush_type.title()])[0] if current_brush_type in TERRAIN_TYPES else current_brush_type.title()}, {current_paint_mode.replace('_',' ')}")
                                break
                        else: # If no mode button clicked, check brush buttons
                            for btn in brush_buttons:
                                if btn['rect'].collidepoint(mouse_pos):
                                    current_brush_type = btn['brush']
                                    set_status(f"Brush: {TERRAIN_TYPES.get(current_brush_type, [current_brush_type.title()])[0] if current_brush_type in TERRAIN_TYPES else current_brush_type.title()}, {current_paint_mode.replace('_',' ')}")
                                    break
                    
                    if clicked_on_ui: continue # Don't process grid click if UI was handled

                    # Process grid clicks
                    clicked_pos_on_grid = get_clicked_grid_pos(mouse_pos, ROWS, grid_pixel_width)
                    if not clicked_pos_on_grid: continue
                    
                    r_click, c_click = clicked_pos_on_grid
                    node_clicked = grid[r_click][c_click]

                    # LEFT MOUSE BUTTON
                    if event.button == 1:
                        drawing_on_grid_mode = True 
                        
                        if current_brush_type == BRUSH_START:
                            if start_node: start_node.reset_to_default_terrain()
                            start_node = node_clicked
                            start_node.make_special_node(BRUSH_START)
                            if end_node == start_node: end_node = None ; grid[r_click][c_click].make_special_node(BRUSH_START) # Ensure start node is set
                        elif current_brush_type == BRUSH_END:
                            if end_node: end_node.reset_to_default_terrain()
                            end_node = node_clicked
                            end_node.make_special_node(BRUSH_END)
                            if start_node == end_node: start_node = None; grid[r_click][c_click].make_special_node(BRUSH_END) # Ensure end node is set
                        
                        elif current_brush_type in TERRAIN_TYPES: # Any terrain brush (Grass, Road, Obstacle, etc.)
                            if current_paint_mode == PAINT_MODE_SINGLE:
                                if node_clicked == start_node: start_node = None
                                if node_clicked == end_node: end_node = None
                                node_clicked.set_terrain(current_brush_type)
                            elif current_paint_mode == PAINT_MODE_FLOOD:
                                run_flood_fill(grid, ROWS, r_click, c_click, current_brush_type, start_node, end_node)
                                drawing_on_grid_mode = False # Flood fill is not a drag operation
                        
                        # H-cost update if start/end changed
                        if start_node and end_node:
                            start_node.h_cost = h(start_node.get_pos(), end_node.get_pos())
                            start_node.f_cost = start_node.g_cost + start_node.h_cost


                    # RIGHT MOUSE BUTTON (Erase to default terrain - always single tile for now)
                    elif event.button == 3:
                        drawing_on_grid_mode = True 
                        if node_clicked == start_node: start_node = None
                        if node_clicked == end_node: end_node = None
                        node_clicked.reset_to_default_terrain()
                
                elif event.type == pygame.MOUSEMOTION:
                    if drawing_on_grid_mode: # Dragging
                        motion_pos_on_grid = get_clicked_grid_pos(mouse_pos, ROWS, grid_pixel_width)
                        if motion_pos_on_grid:
                            r_motion, c_motion = motion_pos_on_grid
                            node_motion = grid[r_motion][c_motion]
                            
                            buttons_pressed = pygame.mouse.get_pressed()
                            if buttons_pressed[0]: # Left mouse drag (always single tile paint)
                                if current_brush_type == BRUSH_START or current_brush_type == BRUSH_END:
                                    pass # Don't drag start/end, only place on click
                                elif current_brush_type in TERRAIN_TYPES:
                                    if node_motion != start_node and node_motion != end_node:
                                        node_motion.set_terrain(current_brush_type)
                            elif buttons_pressed[2]: # Right mouse drag (erase)
                                if node_motion == start_node: start_node = None
                                if node_motion == end_node: end_node = None
                                node_motion.reset_to_default_terrain()

                elif event.type == pygame.MOUSEBUTTONUP:
                    drawing_on_grid_mode = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start_node and end_node and not algorithm_started:
                    algorithm_started = True
                    for r_list in grid:
                        for node_obj in r_list: node_obj.update_neighbors(grid)
                    draw_func = lambda: draw_all(win, grid, ROWS, grid_pixel_width, current_hovered_node, show_instructions, status_message, current_brush_type, current_paint_mode)
                    algorithm(draw_func, grid, start_node, end_node, set_status)
                    algorithm_started = False

                if event.key == pygame.K_c:
                    start_node = None; end_node = None
                    algorithm_started = False; drawing_on_grid_mode = False
                    grid = make_grid(ROWS, grid_pixel_width)
                    set_status("Grid Cleared! Select Brush & Paint Mode.")
                
                if event.key == pygame.K_i: show_instructions = not show_instructions
    pygame.quit()

if __name__ == "__main__":
    main(WIN, TOTAL_WIDTH, GRID_WIDTH)