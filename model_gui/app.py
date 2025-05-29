import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
import chess
import chess.svg
import chess.pgn
from PIL import Image, ImageTk
from cairosvg import svg2png
from io import BytesIO
import numpy as np
import joblib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MoveScorer from models
from models.models import MoveScorer
from visualizer import get_move_arrows
from scripts.utilities.alter_move_prob import alter_move_probabilties
from development.alter_move_prob_train.alter_move_prob_nn import AlterMoveProbNN
from common.board_information import (
    phase_of_game, king_danger, get_lucas_analytics, is_under_mate_threat
)

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Move Scorer Visualiser")
        self.root.geometry("1200x700")
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Default FEN
        self.default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Available weight types
        self.weight_types = ["opening", "midgame", "endgame", "defensive_tactics", "automatic"]
        self.current_weight_type = tk.StringVar(value="midgame")
        
        # For automatic model selection
        self.model_selector_clf = None
        self.auto_selected_model = None
        
        # Mood settings for altering move probabilities
        self.moods = ["confident", "cocky", "cautious", "tilted", "hurry", "flagging", "NN"]
        self.current_mood = tk.StringVar(value="confident")
        self.alter_probs = tk.BooleanVar(value=False)
        self.alter_log = ""
        
        # Neural network model for altering move probabilities
        self.alter_nn_model = None
        
        # Initialize MoveScorer (will be set in setup_gui)
        self.move_scorer = None
        
        # Current board state
        self.current_board = None
        self.current_move_probs = {}
        
        # Flag to prevent recursive analysis
        self.is_analyzing = False
        
        # Flag to prevent recursive navigation
        self.is_navigating = False
        
        # PGN variables
        self.pgn_games = []
        self.current_game = None
        self.game_nodes = []
        self.current_node_idx = 0
        
        # For automatic model selection
        self.model_selector_clf = None
        self.auto_selected_model = None
        
        # Set up the GUI components
        self.setup_gui()
        
        # Initialize the model with default weights
        self.load_model()
        
        # Initial board display
        self.update_board(self.default_fen)
    
    def load_model(self):
        """Load the model with the currently selected weight type"""
        weight_type = self.current_weight_type.get()
        
        # Reset auto selected model
        self.auto_selected_model = None
        
        if weight_type == "automatic":
            ## Load the decision tree classifier if it's not already loaded
            # if self.model_selector_clf is None:
            #     self.load_model_selector_classifier()
            
            # If we have a current board, use it to predict the best model
            if self.current_board is not None:
                # Predict the best model for the current board
                self.auto_selected_model = self.get_game_phase_model(self.current_board)
                weight_type = self.auto_selected_model
                print(f"Automatic mode selected '{weight_type}' model for current position")
            else:
                # Default to midgame if we don't have a board yet
                weight_type = "midgame"
                self.auto_selected_model = weight_type
                print("Automatic mode defaulting to 'midgame' model (no board available)")
        
        move_from_weights_path = f"models/model_weights/piece_selector_{weight_type}_weights.pth"
        move_to_weights_path = f"models/model_weights/piece_to_{weight_type}_weights.pth"
        
        # Initialize MoveScorer with selected weights
        self.move_scorer = MoveScorer(
            move_from_weights_path=move_from_weights_path,
            move_to_weights_path=move_to_weights_path
        )
        
        # Also load the AlterMoveProbNN model if it's not already loaded
        self.load_alter_nn_model()
        
        if self.auto_selected_model:
            self.status_var.set(f"Automatic mode - using {self.auto_selected_model} weights")
        else:
            self.status_var.set(f"Loaded {weight_type} weights")
    
    def load_alter_nn_model(self):
        """Load the AlterMoveProbNN model with its trained weights"""
        if self.alter_nn_model is not None:
            return
        try:
            import torch
            
            # Create a new instance of the model
            self.alter_nn_model = AlterMoveProbNN()
            
            # Load trained weights if they exist
            weights_dir = "development/alter_move_prob_train/data"
            weights_path = f"{weights_dir}/alter_move_prob_nn_best.pth"
            
            # Ensure the directory exists
            os.makedirs(weights_dir, exist_ok=True)
            
            if os.path.exists(weights_path):
                self.alter_nn_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
                print("Loaded AlterMoveProbNN weights from:", weights_path)
            else:
                print("No trained weights found for AlterMoveProbNN. Using default parameters.")
                
            # Set model to evaluation mode
            self.alter_nn_model.eval()
            
        except Exception as e:
            print(f"Error loading AlterMoveProbNN model: {e}")
            import traceback
            traceback.print_exc()
            self.alter_nn_model = None
    
    def setup_gui(self):
        # Top frame for input controls
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill="x")
        
        # Create tabs for FEN input vs PGN import
        self.tab_control = ttk.Notebook(top_frame)
        self.tab_control.pack(fill="x", expand=True)
        
        # FEN Tab
        fen_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(fen_tab, text="FEN Input")
        
        # FEN input
        ttk.Label(fen_tab, text="FEN:").pack(side="left", padx=(0, 10))
        self.fen_var = tk.StringVar(value=self.default_fen)
        fen_entry = ttk.Entry(fen_tab, textvariable=self.fen_var, width=50)
        fen_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        # Analyse button for FEN
        fen_submit_btn = ttk.Button(fen_tab, text="Analyse Position", command=self.on_submit_fen)
        fen_submit_btn.pack(side="left")
        
        # PGN Tab
        pgn_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(pgn_tab, text="PGN Import")
        
        # PGN import button
        pgn_import_btn = ttk.Button(pgn_tab, text="Import PGN", command=self.import_pgn)
        pgn_import_btn.pack(side="left", padx=(0, 10))
        
        # Game selection dropdown
        ttk.Label(pgn_tab, text="Game:").pack(side="left", padx=(10, 5))
        self.game_var = tk.StringVar()
        self.game_dropdown = ttk.Combobox(pgn_tab, textvariable=self.game_var, width=40, state="readonly")
        self.game_dropdown.pack(side="left", padx=(0, 10), fill="x", expand=True)
        self.game_dropdown.bind("<<ComboboxSelected>>", self.on_game_selected)
        
        # Navigation frame for PGN
        nav_frame = ttk.Frame(pgn_tab)
        nav_frame.pack(side="left")
        
        # Navigation buttons
        first_btn = ttk.Button(nav_frame, text="<<", command=lambda: self.safe_navigate("first"))
        first_btn.pack(side="left", padx=2)
        prev_btn = ttk.Button(nav_frame, text="<", command=lambda: self.safe_navigate("prev"))
        prev_btn.pack(side="left", padx=2)
        next_btn = ttk.Button(nav_frame, text=">", command=lambda: self.safe_navigate("next"))
        next_btn.pack(side="left", padx=2)
        last_btn = ttk.Button(nav_frame, text=">>", command=lambda: self.safe_navigate("last"))
        last_btn.pack(side="left", padx=2)
        
        # Move counter label
        self.move_counter_var = tk.StringVar(value="Move: 0/0")
        move_counter = ttk.Label(nav_frame, textvariable=self.move_counter_var)
        move_counter.pack(side="left", padx=(10, 0))
        
        # Weight type selection (outside tabs, affects both modes)
        weight_frame = ttk.Frame(top_frame)
        weight_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(weight_frame, text="Weight Type:").pack(side="left", padx=(0, 5))
        weight_combo = ttk.Combobox(weight_frame, textvariable=self.current_weight_type, 
                                   values=self.weight_types, width=10, state="readonly")
        weight_combo.pack(side="left", padx=(0, 10))
        weight_combo.bind("<<ComboboxSelected>>", self.on_weight_changed)
        
        # Label to display the auto-selected model
        self.auto_model_label_var = tk.StringVar(value="")
        self.auto_model_label = ttk.Label(weight_frame, textvariable=self.auto_model_label_var)
        self.auto_model_label.pack(side="left", padx=(0, 10))
        
        # Alter probability controls
        alter_frame = ttk.Frame(top_frame)
        alter_frame.pack(fill="x", pady=(10, 0))
        
        # Tickbox for enabling altered probabilities
        self.alter_check = ttk.Checkbutton(alter_frame, text="Alter Move Probabilities", 
                                         variable=self.alter_probs, command=self.on_alter_changed)
        self.alter_check.pack(side="left", padx=(0, 10))
        
        # Mood selection dropdown
        ttk.Label(alter_frame, text="Mood:").pack(side="left", padx=(0, 5))
        mood_combo = ttk.Combobox(alter_frame, textvariable=self.current_mood, 
                                 values=self.moods, width=10, state="readonly")
        mood_combo.pack(side="left")
        mood_combo.bind("<<ComboboxSelected>>", self.on_mood_changed)
        
        # Main content frame
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill="both", expand=True)
        
        # Configure grid for resizable panels
        content_frame.columnconfigure(0, weight=3)  # Chess board column
        content_frame.columnconfigure(1, weight=2)  # Moves column
        content_frame.columnconfigure(2, weight=2)  # Log column
        content_frame.columnconfigure(3, weight=2)  # Game moves column
        content_frame.rowconfigure(0, weight=1)     # All panels in one row
        
        # Left side: Chess board
        self.board_frame = ttk.LabelFrame(content_frame, text="Chess Board")
        self.board_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.board_frame.rowconfigure(0, weight=1)
        self.board_frame.columnconfigure(0, weight=1)
        
        self.board_canvas = tk.Canvas(self.board_frame, bg="white")
        self.board_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Controls below the board
        board_controls = ttk.Frame(self.board_frame)
        board_controls.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Label(board_controls, text="Show arrows for top moves:").pack(side="left", padx=(0, 5))
        self.arrow_var = tk.BooleanVar(value=True)
        arrow_check = ttk.Checkbutton(board_controls, variable=self.arrow_var, command=self.redraw_board)
        arrow_check.pack(side="left", padx=(0, 10))
        
        ttk.Label(board_controls, text="Number of arrows:").pack(side="left", padx=(0, 5))
        self.arrow_count_var = tk.IntVar(value=3)
        arrow_count = ttk.Spinbox(board_controls, from_=1, to=10, width=3, textvariable=self.arrow_count_var, 
                                 command=self.redraw_board, wrap=True)
        arrow_count.pack(side="left")
        
        # Right side: Move probabilities
        moves_frame = ttk.LabelFrame(content_frame, text="Human-like Move Probabilities")
        moves_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        moves_frame.rowconfigure(0, weight=1)
        moves_frame.columnconfigure(0, weight=1)
        
        # Scrollable frame for moves
        self.scrollable_frame = ttk.Frame(moves_frame)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.scrollable_frame.rowconfigure(0, weight=1)
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # Move list with probabilities
        columns = ("Rank", "Move", "Probability")
        self.moves_tree = ttk.Treeview(self.scrollable_frame, columns=columns, show="headings")
        self.moves_tree.heading("Rank", text="#")
        self.moves_tree.heading("Move", text="Move")
        self.moves_tree.heading("Probability", text="Probability")
        self.moves_tree.column("Rank", width=40)
        self.moves_tree.column("Move", width=100)
        self.moves_tree.column("Probability", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.scrollable_frame, orient="vertical", command=self.moves_tree.yview)
        self.moves_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.moves_tree.grid(row=0, column=0, sticky="nsew")
        
        # Bind event to highlight the move on the board when selected
        self.moves_tree.bind("<<TreeviewSelect>>", self.on_move_selected)
        
        # Log frame for displaying alteration log
        log_frame = ttk.LabelFrame(content_frame, text="Probability Alteration Log")
        log_frame.grid(row=0, column=2, sticky="nsew", padx=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        # Text widget for displaying the log
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add scrollbar for log
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Game moves panel
        game_moves_frame = ttk.LabelFrame(content_frame, text="Game Moves")
        game_moves_frame.grid(row=0, column=3, sticky="nsew", padx=5)
        game_moves_frame.rowconfigure(0, weight=1)
        game_moves_frame.columnconfigure(0, weight=1)
        
        # Scrollable frame for game moves
        self.game_moves_frame = ttk.Frame(game_moves_frame)
        self.game_moves_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.game_moves_frame.rowconfigure(0, weight=1)
        self.game_moves_frame.columnconfigure(0, weight=1)
        
        # Game moves list
        columns = ("Move Number", "White", "Black")
        self.game_moves_tree = ttk.Treeview(self.game_moves_frame, columns=columns, show="headings")
        self.game_moves_tree.heading("Move Number", text="#")
        self.game_moves_tree.heading("White", text="White")
        self.game_moves_tree.heading("Black", text="Black")
        self.game_moves_tree.column("Move Number", width=40)
        self.game_moves_tree.column("White", width=80)
        self.game_moves_tree.column("Black", width=80)
        
        # Add scrollbar for game moves
        game_scrollbar = ttk.Scrollbar(self.game_moves_frame, orient="vertical", command=self.game_moves_tree.yview)
        self.game_moves_tree.configure(yscrollcommand=game_scrollbar.set)
        game_scrollbar.grid(row=0, column=1, sticky="ns")
        self.game_moves_tree.grid(row=0, column=0, sticky="nsew")
        
        # Bind event to navigate to move when selected
        self.game_moves_tree.bind("<<TreeviewSelect>>", self.on_game_move_selected)
        
        # Bind canvas resize event
        self.board_canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(fill="x", side="bottom", pady=(5, 0))
    
    def import_pgn(self):
        """Import a PGN file"""
        filepath = filedialog.askopenfilename(
            title="Select PGN File",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Clear any existing games
            self.pgn_games = []
            self.game_nodes = []
            
            # Read all games from the PGN file
            with open(filepath, 'r', encoding='utf-8-sig') as pgn_file:
                game = chess.pgn.read_game(pgn_file)
                game_count = 0
                
                while game is not None:
                    game_count += 1
                    
                    # Construct game description
                    white = game.headers.get("White", "Unknown")
                    black = game.headers.get("Black", "Unknown")
                    date = game.headers.get("Date", "????-??-??")
                    result = game.headers.get("Result", "*")
                    
                    description = f"Game {game_count}: {white} vs {black} ({date}) {result}"
                    self.pgn_games.append((description, game))
                    
                    # Read next game
                    game = chess.pgn.read_game(pgn_file)
            
            # Update the game dropdown
            self.game_dropdown['values'] = [game[0] for game in self.pgn_games]
            
            # If games were loaded, select the first one
            if self.pgn_games:
                self.game_dropdown.current(0)
                self.on_game_selected(None)
                self.status_var.set(f"Loaded PGN with {len(self.pgn_games)} games")
            else:
                self.status_var.set("No games found in the PGN file")
        
        except Exception as e:
            self.status_var.set(f"Error loading PGN: {str(e)}")
            print(f"Error loading PGN: {e}")
    
    def on_game_selected(self, event):
        """Handle game selection from dropdown"""
        if not self.pgn_games:
            return
        
        # Get the selected game index
        selected_idx = self.game_dropdown.current()
        if selected_idx < 0:
            return
        
        # Set the current game
        self.current_game = self.pgn_games[selected_idx][1]
        
        # Generate move list for navigation
        self.game_nodes = [self.current_game]
        node = self.current_game
        
        # Print debug info
        print(f"Generating move list from game with {len(node.variations)} variations")
        
        while node.variations:
            node = node.variations[0]  # Main line
            self.game_nodes.append(node)
            # Print debug info for each node
            print(f"Added node: move={node.move}, ply={node.ply()}, san={node.san() if node.move else 'None'}")
        
        print(f"Total nodes in game: {len(self.game_nodes)}")
        
        # Clear current move list
        for item in self.game_moves_tree.get_children():
            self.game_moves_tree.delete(item)
        
        # Populate the move list
        move_pairs = {}
        current_move_num = 0
        
        for node in self.game_nodes[1:]:  # Skip the root node
            prev_node = node.parent
            board = prev_node.board()
            
            if board.turn == chess.WHITE:
                current_move_num += 1
                move_pairs[current_move_num] = [node.san(), ""]
            else:
                move_pairs[current_move_num][1] = node.san()
        
        # Add move pairs to the treeview
        for move_num, moves in move_pairs.items():
            self.game_moves_tree.insert("", "end", values=(move_num, moves[0], moves[1]), tags=(str(move_num),))
        
        # Start from the beginning of the game
        self.current_node_idx = 0
        self.safe_navigate("first")

    def safe_navigate(self, direction):
        """Safe wrapper for navigate_game to handle errors gracefully"""
        try:
            self.navigate_game(direction)
        except Exception as e:
            print(f"Error during navigation: {e}")
            import traceback
            traceback.print_exc()
            
    def navigate_game(self, direction):
        """Navigate through the current game"""
        if not self.game_nodes:
            return
            
        # Prevent recursive navigation
        if self.is_navigating:
            print("Already navigating, ignoring request")
            return
            
        self.is_navigating = True
        
        try:
            # Store the navigation state
            old_idx = self.current_node_idx
            new_idx = old_idx  # Initialize new index to old index
            total_nodes = len(self.game_nodes)
            
            print(f"Navigation requested: direction={direction}, current_idx={old_idx}, total_nodes={total_nodes}")
            
            if direction == "first":
                new_idx = 0
            elif direction == "prev":
                new_idx = max(0, old_idx - 1)
            elif direction == "next":
                # Ensure we can go to the next node
                target_idx = old_idx + 1
                print(f"Attempting to navigate to next move: {target_idx}")
                
                if target_idx < total_nodes:
                    new_idx = target_idx
                    print(f"Moving to node {new_idx}")
                else:
                    print(f"Cannot navigate: already at last node ({old_idx} of {total_nodes-1})")
                    new_idx = old_idx  # Stay at current position
            elif direction == "last":
                new_idx = total_nodes - 1
            elif isinstance(direction, int):
                new_idx = min(max(0, direction), total_nodes - 1)
                print(f"Navigating to specific index: {new_idx}")
            
            # Update the current index
            self.current_node_idx = new_idx
            
            # Only update if the position changed
            if old_idx != new_idx:
                node = self.game_nodes[new_idx]
                
                # Debug info about the node
                if node.parent:
                    print(f"Node {new_idx}: move={node.move}, san={node.san()}, ply={node.ply()}")
                else:
                    print(f"Node {new_idx}: Root node")
                    
                board = node.board()
                
                # Update the current board and display
                self.update_board(board.fen())
                
                # Update move counter
                self.move_counter_var.set(f"Move: {new_idx}/{total_nodes - 1}")
                
                # Select the corresponding move in the game moves tree
                self.select_current_move_in_tree()
            else:
                print(f"No navigation performed: remained at node {old_idx}")
        finally:
            self.is_navigating = False
    
    def select_current_move_in_tree(self):
        """Select the current move in the game moves tree"""
        if self.current_node_idx <= 0 or self.is_navigating:
            return
            
        try:
            # Temporarily unbind the selection event to prevent recursive calls
            self.game_moves_tree.unbind("<<TreeviewSelect>>")
            
            node = self.game_nodes[self.current_node_idx]
            
            # Calculate the move number and whether it's a white or black move
            ply = node.ply()
            move_number = (ply + 1) // 2  # 1-based move number
            is_black_move = (ply % 2 == 0)  # Even ply means black's move
            
            print(f"Selecting move in tree: node={self.current_node_idx}, move_number={move_number}, {'black' if is_black_move else 'white'}, ply={ply}")
            
            # Clear current selection
            for item_id in self.game_moves_tree.selection():
                self.game_moves_tree.selection_remove(item_id)
            
            # Find and select the corresponding item in the tree
            for item_id in self.game_moves_tree.get_children():
                item = self.game_moves_tree.item(item_id)
                tree_move_number = int(item['values'][0])
                
                if tree_move_number == move_number:
                    # Select this row in the tree view
                    self.game_moves_tree.selection_set(item_id)
                    self.game_moves_tree.see(item_id)
                    print(f"Selected move number {move_number} in tree")
                    break
            else:
                print(f"Could not find move number {move_number} in tree")
                
        except Exception as e:
            print(f"Error selecting move in tree: {e}")
        finally:
            # Rebind the selection event
            self.game_moves_tree.bind("<<TreeviewSelect>>", self.on_game_move_selected)
    
    def on_game_move_selected(self, event):
        """Handle move selection from the game moves tree"""
        # Ignore if we're already navigating
        if self.is_navigating:
            return
            
        selection = self.game_moves_tree.selection()
        if not selection:
            return
        
        # Get the move number
        item = self.game_moves_tree.item(selection[0])
        move_number = int(item['values'][0])
        
        # Check if it's a white or black move
        col = self.game_moves_tree.identify_column(event.x)
        is_black_move = (col == '#3')  # Black's move is in column 3
        
        print(f"Move selected: {move_number} {'black' if is_black_move else 'white'}")
        
        # Find the corresponding node in the game tree
        target_node_idx = None
        
        for i, node in enumerate(self.game_nodes):
            if i == 0:  # Skip root node
                continue
                
            # Calculate the move number for this node based on its ply
            ply = node.ply()
            node_move_number = (ply + 1) // 2  # 1-based move number
            node_is_black = (ply % 2 == 0)  # Even ply means black's move
            
            print(f"Checking node {i}: move_number={node_move_number}, {'black' if node_is_black else 'white'}, ply={ply}")
            
            if node_move_number == move_number and node_is_black == is_black_move:
                target_node_idx = i
                print(f"Found matching node at index {target_node_idx}")
                break
        
        # Navigate to that position (directly, not via next)
        if target_node_idx is not None and 0 <= target_node_idx < len(self.game_nodes):
            print(f"Directly navigating to node index {target_node_idx}")
            self.safe_navigate(target_node_idx)
        else:
            print(f"Could not find node for move {move_number} {'black' if is_black_move else 'white'}")
    
    def on_weight_changed(self, event):
        """Handle weight type change"""
        try:
            self.load_model()
            # Re-analyse with new model if a board is loaded
            if self.current_board:
                self.analyse_position(self.current_board)
        except Exception as e:
            self.status_var.set(f"Error loading weights: {str(e)}")
            print(f"Error loading weights: {e}")
    
    def redraw_board(self):
        if self.current_board is not None:
            self.render_board()
    
    def render_board(self):
        try:
            # Get current canvas size
            canvas_width = self.board_canvas.winfo_width()
            canvas_height = self.board_canvas.winfo_height()
            
            # Ensure we have valid dimensions
            if canvas_width < 10 or canvas_height < 10:
                # Canvas not properly sized yet, use default size
                board_size = 400
            else:
                # Use the smaller dimension to ensure a square board
                board_size = min(canvas_width, canvas_height)
            
            # Create SVG with or without arrows
            if self.arrow_var.get() and self.current_move_probs:
                # Get arrows for the most probable moves
                try:
                    arrows = get_move_arrows(self.current_board, self.current_move_probs, self.arrow_count_var.get())
                    print(f"Arrows generated: {arrows}")
                    
                    # Generate SVG with arrows
                    svg_data = chess.svg.board(
                        board=self.current_board,
                        size=board_size,
                        arrows=arrows,
                        coordinates=True
                    )
                except Exception as arrow_error:
                    print(f"Error generating arrows: {arrow_error}")
                    # Fallback to board without arrows
                    svg_data = chess.svg.board(board=self.current_board, size=board_size, coordinates=True)
            else:
                # Generate SVG without arrows
                svg_data = chess.svg.board(board=self.current_board, size=board_size, coordinates=True)
            
            # Convert SVG to PNG
            png_data = svg2png(bytestring=svg_data)
            img_data = BytesIO(png_data)
            
            # Create an image from the PNG data
            img = Image.open(img_data)
            self.tk_img = ImageTk.PhotoImage(img)
            
            # Update the canvas
            self.board_canvas.delete("all")
            # Center the image if canvas is larger than the board
            x_offset = max(0, (canvas_width - board_size) // 2)
            y_offset = max(0, (canvas_height - board_size) // 2)
            self.board_canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_img)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error rendering board: {str(e)}")
            print(f"Error rendering board: {e}")
    
    def update_board(self, fen):
        try:
            board = chess.Board(fen)
            self.current_board = board
            
            # Render the board
            self.render_board()
            
            # Update status
            self.status_var.set(f"Position loaded: {fen}")
            
            # If we're in automatic mode, we need to reload the model
            # for the new position before analyzing
            if self.current_weight_type.get() == "automatic":
                self.load_model()
            
            # Analyse the position if valid and not already analyzing
            if not self.is_analyzing:
                self.analyse_position(board)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error: {e}")
    
    def analyse_position(self, board):
        # Set analyzing flag to prevent recursive calls
        if self.is_analyzing:
            return
            
        self.is_analyzing = True
        
        try:
            # Clear previous move list
            for item in self.moves_tree.get_children():
                self.moves_tree.delete(item)
            
            # Reset move probabilities and log
            self.current_move_probs = {}
            self.alter_log = ""
            
            # Clear log display
            self.update_log_display("")
            
            # If using automatic mode, select the appropriate model for this position
            if self.current_weight_type.get() == "automatic":
                # # Ensure the classifier is loaded
                # if self.model_selector_clf is None:
                #     self.load_decision_tree_classifier()
                
                # Predict the best model for this position
                selected_model = self.get_game_phase_model(board)
                
                # Update the auto selected model
                self.auto_selected_model = selected_model
                
                # Update the label
                self.auto_model_label_var.set(f"(using {selected_model} model)")
                
                # Load the selected model
                self.load_model()
            else:
                # Clear the auto model label if not in automatic mode
                self.auto_model_label_var.set("")
            
            # Store original board for display
            original_board = board.copy()
            
            # If board is black to move, we need to mirror it as MoveScorer requires white to move
            mirrored = False
            if board.turn == chess.BLACK:
                move_scorer_board = board.mirror()
                mirrored = True
                self.status_var.set(self.status_var.get() + " (Board mirrored for analysis as black to move)")
            else:
                move_scorer_board = board.copy()
            
            # Get moves and probabilities
            _, move_probs = self.move_scorer.get_move_dic(move_scorer_board, san=True, top=20)

            # If we mirrored the board, we need to convert the moves back to the original board perspective
            if mirrored:
                # For display purposes, keep the original board
                self.current_board = original_board
                
                # Convert moves from mirrored board to original board perspective
                original_move_probs = {}
                for move_san, prob in move_probs.items():
                    # Try to map the move back to the original board
                    try:
                        # Parse the move on the mirrored board
                        move = move_scorer_board.parse_san(move_san)
                        # Mirror the move
                        mirrored_move = chess.Move(
                            chess.square_mirror(move.from_square),
                            chess.square_mirror(move.to_square),
                            move.promotion
                        )
                        # Get the SAN for the mirrored move on the original board
                        if mirrored_move in original_board.legal_moves:
                            original_move_san = original_board.san(mirrored_move)
                            original_move_probs[original_move_san] = prob
                    except ValueError:
                        continue
                
                move_probs = original_move_probs.copy()
            
            # Apply alteration if enabled
            if self.alter_probs.get():
                # We need UCI moves for alteration
                uci_probs = {}
                for move_san, prob in move_probs.items():
                    try:
                        move = board.parse_san(move_san)
                        uci_probs[move.uci()] = prob
                    except ValueError:
                        print(f"Error parsing move: {move_san} in position {board.fen()}")
                        continue
                
                # Get previous positions if available
                prev_board = None
                prev_prev_board = None
                
                # If we're in a game, we might be able to get previous positions
                if self.current_node_idx > 0 and self.game_nodes:
                    curr_node = self.game_nodes[self.current_node_idx]
                    # prev_board is the board 1 ply ago (opponent's move)
                    if curr_node.parent:
                        prev_board = curr_node.parent.board()
                        # prev_board = prev_board.mirror()
                    # prev_prev_board is the board 2 plies ago (our previous move)
                    if prev_board and curr_node.parent.parent:
                        prev_prev_board = curr_node.parent.parent.board()
                        # prev_prev_board = prev_prev_board.mirror()
                        
                # If we mirrored the current board (black to move in original position),
                # we need to mirror the previous boards too
                # if mirrored and prev_board:
                #     prev_board = prev_board.mirror()
                #     print(f"Mirrored prev_board: {prev_board.fen()}")
                
                # if mirrored and prev_prev_board:
                #     prev_prev_board = prev_prev_board.mirror()
                #     print(f"Mirrored prev_prev_board: {prev_prev_board.fen()}")
                
                print(f"Current board: {board.fen()}")
                if prev_board:
                    print(f"Previous board: {prev_board.fen()}")
                if prev_prev_board:
                    print(f"Previous previous board: {prev_prev_board.fen()}")
                
                # Print original move probabilities for debugging
                print("Original UCI probabilities:")
                sorted_uci_probs = sorted(uci_probs.items(), key=lambda x: x[1], reverse=True)
                for move_uci, prob in sorted_uci_probs[:5]:  # Print top 5 moves
                    print(f"{move_uci}: {prob:.4f}")
                
                # Alter the move probabilities
                mood = self.current_mood.get()
                
                # Use neural network model if mood is set to "NN"
                if mood == "NN" and self.alter_nn_model is not None:
                    import torch
                    
                    # Convert dictionary to torch tensors
                    with torch.no_grad():
                        try:
                            # Call the NN model's forward function
                            altered_probs, self.alter_log = self.alter_nn_model(
                                uci_probs, board, prev_board, prev_prev_board
                            )
                            
                            # Convert any tensor values back to Python floats
                            altered_probs = {k: float(v) if torch.is_tensor(v) else v 
                                            for k, v in altered_probs.items()}
                                            
                        except Exception as e:
                            print(f"Error using AlterMoveProbNN: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fall back to unaltered probabilities
                            altered_probs = uci_probs
                            self.alter_log = f"Error using AlterMoveProbNN: {str(e)}"
                else:
                    # Use regular mood-based alteration
                    altered_probs, self.alter_log = alter_move_probabilties(
                        uci_probs, board, prev_board, prev_prev_board, mood
                    )
                
                # Print altered move probabilities for debugging
                print("Altered UCI probabilities:")
                sorted_altered_probs = sorted(altered_probs.items(), key=lambda x: x[1], reverse=True)
                for move_uci, prob in sorted_altered_probs[:5]:  # Print top 5 moves
                    print(f"{move_uci}: {prob:.4f}")
                
                # Convert back to SAN
                move_probs = {}
                for move_uci, prob in altered_probs.items():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            move_probs[board.san(move)] = prob
                    except ValueError:
                        continue
                
                # Update log display
                self.update_log_display(self.alter_log)
            
            # Store current move probabilities for visualization
            self.current_move_probs = move_probs
            
            # Display moves and probabilities
            for i, (move, prob) in enumerate(sorted(move_probs.items(), key=lambda x: x[1], reverse=True), 1):
                # Format probability as percentage
                prob_str = f"{prob*100:.2f}%"
                self.moves_tree.insert("", "end", values=(i, move, prob_str), tags=(move,))
            
            # Re-render board with arrows
            self.render_board()
                
        except Exception as e:
            self.status_var.set(f"Analysis error: {str(e)}")
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always reset analyzing flag when done
            self.is_analyzing = False
    
    def on_move_selected(self, event):
        # Get selected item
        selection = self.moves_tree.selection()
        if not selection:
            return
        
        item = self.moves_tree.item(selection[0])
        move_san = item['values'][1]  # Get the move in SAN notation
        
        # Highlight this move on the board
        try:
            move = self.current_board.parse_san(move_san)
            arrows = [chess.svg.Arrow(move.from_square, move.to_square, color="#00FF00")]  # Green arrow for selected move
            print(f"Selected move arrows: {arrows}")
            
            try:
                # Get current canvas size
                canvas_width = self.board_canvas.winfo_width()
                canvas_height = self.board_canvas.winfo_height()
                
                # Ensure we have valid dimensions
                if canvas_width < 10 or canvas_height < 10:
                    # Canvas not properly sized yet, use default size
                    board_size = 400
                else:
                    # Use the smaller dimension to ensure a square board
                    board_size = min(canvas_width, canvas_height)
                
                # Generate SVG with just this move highlighted
                svg_data = chess.svg.board(
                    board=self.current_board,
                    size=board_size,
                    arrows=arrows,
                    coordinates=True
                )
                
                # Convert SVG to PNG
                png_data = svg2png(bytestring=svg_data)
                img_data = BytesIO(png_data)
                
                # Create an image from the PNG data
                img = Image.open(img_data)
                self.tk_img = ImageTk.PhotoImage(img)
                
                # Update the canvas
                self.board_canvas.delete("all")
                # Center the image if canvas is larger than the board
                x_offset = max(0, (canvas_width - board_size) // 2)
                y_offset = max(0, (canvas_height - board_size) // 2)
                self.board_canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_img)
                
                self.status_var.set(f"Selected move: {move_san}")
            except Exception as svg_error:
                import traceback
                traceback.print_exc()
                print(f"Error creating SVG/PNG: {svg_error}")
                # Fall back to regular board
                self.redraw_board()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error highlighting move: {e}")
    
    def on_submit_fen(self):
        fen = self.fen_var.get()
        self.update_board(fen)
    
    def on_mood_changed(self, event):
        """Handle mood selection change"""
        # Check if we need to load/reload the NN model
        if self.current_mood.get() == "NN" and self.alter_nn_model is None:
            self.load_alter_nn_model()
            
        # Only need to re-analyze if alter probabilities is turned on
        if self.alter_probs.get() and self.current_board:
            self.analyse_position(self.current_board)
    
    def on_alter_changed(self):
        """Handle toggling of altering probabilities"""
        if self.current_board:
            self.analyse_position(self.current_board)
    
    def update_log_display(self, log_text):
        """Update the log display with provided text"""
        # Enable the text widget for editing
        self.log_text.config(state=tk.NORMAL)
        # Clear current content
        self.log_text.delete(1.0, tk.END)
        # Insert new log text
        self.log_text.insert(tk.END, log_text)
        # Disable editing again
        self.log_text.config(state=tk.DISABLED)

    def get_game_phase_model(self, board):
        """
        Load a vanilla classifier for automatic model selection.
        The classifier just obtains the game phase of the board and chooses the
        corresponding model. In the event that a mate threat is present, the
        classifier will choose the defensive tactics model.
        """
        game_phase = phase_of_game(board)
        if is_under_mate_threat(board, board.turn):
            return "defensive_tactics"
        else:
            return game_phase
        
    def load_decision_tree_classifier(self):
        """
        Load the decision tree classifier for automatic model selection.
        If the model doesn't exist, train a new one.
        """
        model_path = 'development/alter_move_prob_train/data/model_selector_clf.joblib'
        
        try:
            if os.path.exists(model_path):
                # Load the existing model
                self.model_selector_clf = joblib.load(model_path)
                print(f"Loaded model selector from {model_path}")
            else:
                print(f"Model selector not found at {model_path}. Creating a new one...")
                # Create output directory if it doesn't exist
                os.makedirs('development/alter_move_prob_train/data', exist_ok=True)
                
                # Train a new model using the function from create_data.py
                self.model_selector_clf = self._train_decision_tree_classifier()
                
                # Save the classifier for future use
                joblib.dump(self.model_selector_clf, model_path)
                print(f"Saved model selector to {model_path}")
                
        except Exception as e:
            print(f"Error loading model selector: {e}")
            import traceback
            traceback.print_exc()
            self.model_selector_clf = None
    
    # def _train_decision_tree_classifier(self):
    #     """
    #     Train a decision tree classifier based on analysis_results.csv.
    #     This is a copy of the function from create_data.py.
    #     """
    #     from sklearn.tree import DecisionTreeClassifier
    #     import pandas as pd
        
    #     try:
    #         # Load the analysis results data
    #         df = pd.read_csv('model_research/results/analysis_results.csv')
            
    #         # Prepare the features and target
    #         X = df[['complexity', 'win_prob', 'efficient_mobility', 'narrowness', 
    #                 'piece_activity', 'game_phase', 'self_king_danger', 'opp_king_danger']].copy()
            
    #         # Convert game_phase to numeric
    #         phase_map = {'opening': 0, 'midgame': 1, 'endgame': 2}
    #         X['game_phase'] = X['game_phase'].map(phase_map)
            
    #         # Get the best model for each position (minimum rank)
    #         rank_columns = ['opening_rank', 'midgame_rank', 'endgame_rank', 
    #                        'tactics_rank', 'defensive_tactics_rank']
    #         df['best_model'] = df[rank_columns].idxmin(axis=1)
    #         df['best_model'] = df['best_model'].apply(lambda x: x.replace('_rank', ''))
    #         y = df['best_model']
            
    #         # Train the decision tree
    #         clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42)
    #         clf.fit(X, y)
            
    #         return clf
            
    #     except Exception as e:
    #         print(f"Error training decision tree classifier: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
            
    def predict_best_model(self, board):
        """
        Predict the best move scorer model to use for a given board position.
        This is a copy of the function from create_data.py.
        """
        if self.model_selector_clf is None:
            # If no classifier is loaded, default to midgame
            return "midgame"
            
        try:
            # Get game phase
            game_phase = phase_of_game(board)
            phase_map = {'opening': 0, 'midgame': 1, 'endgame': 2}
            game_phase_numeric = phase_map[game_phase]
            
            # Get king danger values
            self_king_danger = king_danger(board, board.turn, game_phase)
            opp_king_danger = king_danger(board, not board.turn, game_phase)
            
            # Get Lucas analytics
            complexity, win_prob, efficient_mobility, narrowness, piece_activity = get_lucas_analytics(board)
            
            # Create a feature vector for prediction
            import pandas as pd
            features = pd.DataFrame([[
                complexity, win_prob, efficient_mobility, narrowness, 
                piece_activity, game_phase_numeric, self_king_danger, opp_king_danger
            ]], columns=['complexity', 'win_prob', 'efficient_mobility', 'narrowness', 
                         'piece_activity', 'game_phase', 'self_king_danger', 'opp_king_danger'])
            
            # Predict the best model
            model_name = self.model_selector_clf.predict(features)[0]
            
            return model_name
            
        except Exception as e:
            print(f"Error predicting best model: {e}")
            import traceback
            traceback.print_exc()
            # Default to midgame if there's an error
            return "midgame"

    def on_canvas_resize(self, event):
        self.render_board()

def main():
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 