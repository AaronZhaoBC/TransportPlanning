import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI

DEFAULT_OPENAI_KEY = (
    "sk-proj-bhcLSdjLUnlP-d3POL6r6X8Rjys5EEULMkCt-PFwVgHhmF27cxWLwxlVEbTDNFw_"
    "DpXnipBckXT3BlbkFJXvE-RJPH6KhxCtIxE7rH1wJ_D_YedoMFANa9aPk-p74SORUqLEliN"
    "GpHh6Td6LqXbW5KpBrS4A"
)


@dataclass
class HistoricalStats:
    load_groups: Dict[str, List[str]]  # LOAD_NO -> list of ITEM_NOs
    combo_frequency: Counter
    pair_frequency: Counter
    item_attributes: Dict[str, Dict[str, Any]]  # ITEM_NO -> attributes dict
    load_details: Dict[str, Dict[str, Any]]  # LOAD_NO -> complete load details


class SteelLoadingPlanner:
    """
    Learn historical steel loading patterns and generate new loading plans with OpenAI assistance.

    The planner uses the history stored in data.xlsx to derive frequent loading templates.
    When new steel items arrive, it can propose a baseline plan based on historical combinations
    and optionally refine it via an OpenAI model.
    """

    def __init__(
        self,
        data_path: str,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        top_k_combos: int = 10,
    ) -> None:
        self.data_path = data_path
        self.model = model
        self.top_k_combos = top_k_combos
        self.dataframe = self._load_data()
        self.history = self._build_history()
        self.client = self._init_openai_client(openai_api_key)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.data_path)
        required_columns = {"ORDER_NO", "LOAD_NO"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"The dataset must contain the columns {required_columns}. Missing: {missing}"
            )

        df["ORDER_NO"] = df["ORDER_NO"].astype(str).str.strip()
        df["LOAD_NO"] = df["LOAD_NO"].astype(str).str.strip()
        
        # Keep physical attribute columns if they exist
        attribute_columns = [
            "PROJECTID", "CUSTOMERID", "VEHICLE_TYPE", "POSTAL_SECTOR", "WEIGHT",
             "PIECES", "REQ_DATE", "LOAD_NO"
        ]
        return df

    def _build_history(self) -> HistoricalStats:
        grouped = self.dataframe.groupby("LOAD_NO")["ORDER_NO"].apply(list)
        load_groups = grouped.to_dict()

        combo_counter: Counter = Counter()
        pair_counter: Counter = Counter()
        
        # Extract physical attributes for each ORDER_NO
        item_attributes: Dict[str, Dict[str, Any]] = {}
        attribute_columns = [
            "PROJECTID", "CUSTOMERID", "VEHICLE_TYPE", "POSTAL_SECTOR", "WEIGHT",
             "PIECES", "REQ_DATE", "LOAD_NO"
        ]
        
        for item_no in self.dataframe["ORDER_NO"].unique():
            item_data = self.dataframe[self.dataframe["ORDER_NO"] == item_no].iloc[0]
            attrs: Dict[str, Any] = {}
            for col in attribute_columns:
                if col in self.dataframe.columns:
                    value = item_data[col]
                    if pd.notna(value):
                        if col in ["LENGTH", "WIDTH", "HEIGHT", "DIAMETER", "WEIGHT"]:
                            try:
                                attrs[col] = float(value)
                            except (ValueError, TypeError):
                                attrs[col] = value
                        elif col == "PIECES":
                            try:
                                attrs[col] = int(float(value))
                            except (ValueError, TypeError):
                                attrs[col] = value
                        elif col == "POSTA_SECTOR":
                            try:
                                attrs[col] = int(float(value))
                            except (ValueError, TypeError):
                                attrs[col] = value
                        else:
                            attrs[col] = str(value)
                    else:
                        attrs[col] = None
            item_attributes[item_no] = attrs

        # Build detailed load information for each LOAD_NO
        load_details: Dict[str, Dict[str, Any]] = {}
        for load_no, items in load_groups.items():
            load_df = self.dataframe[self.dataframe["LOAD_NO"] == load_no]
            
            # Get all items in this load
            unique_items = list(dict.fromkeys(items))
            
            # Calculate aggregate statistics
            total_weight_kg = 0.0
            total_weight_ton = 0.0
            total_pieces = 0
            
            item_details = []
            for item_no in unique_items:
                item_rows = load_df[load_df["ORDER_NO"] == item_no]
                if len(item_rows) > 0:
                    item_data = item_rows.iloc[0]
                    item_attrs = {}
                    for col in attribute_columns:
                        if col in self.dataframe.columns:
                            value = item_data[col]
                            if pd.notna(value):
                                try:
                                    if col == "WEIGHT":
                                        ton_value = float(value)
                                        item_attrs[col] = ton_value
                                        total_weight_ton += ton_value
                                        total_weight_kg += ton_value * 1000
                                    elif col == "PIECES":
                                        pieces_value = int(float(value))
                                        item_attrs[col] = pieces_value
                                        total_pieces += pieces_value
                                    elif col == "PIECES":
                                        item_attrs[col] = int(float(value))
                                    else:
                                        item_attrs[col] = str(value)
                                except (ValueError, TypeError):
                                    item_attrs[col] = value
                    item_details.append({"ORDER_NO": item_no, "attributes": item_attrs})
            
            load_details[load_no] = {
                "items": unique_items,
                "item_set": set(unique_items),
                "item_count": len(unique_items),
                "total_weight_kg": total_weight_kg,
                "total_weight_ton": total_weight_ton,
                "total_pieces": total_pieces,
                "item_details": item_details,
            }

        for items in grouped:
            unique_items = list(dict.fromkeys(items))  # preserve order, drop duplicates
            combo_counter[frozenset(unique_items)] += 1

            for i in range(len(unique_items)):
                for j in range(i + 1, len(unique_items)):
                    pair = frozenset({unique_items[i], unique_items[j]})
                    pair_counter[pair] += 1

        return HistoricalStats(
            load_groups=load_groups,
            combo_frequency=combo_counter,
            pair_frequency=pair_counter,
            item_attributes=item_attributes,
            load_details=load_details,
        )

    def _init_openai_client(self, api_key: Optional[str]) -> Optional[OpenAI]:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_KEY
        if not resolved_key:
            return None
        return OpenAI(api_key=resolved_key)

    def _build_context_summary(self) -> str:
        total_loads = len(self.history.load_groups)
        unique_items = self.dataframe["ORDER_NO"].nunique()
        summary_lines = [
            f"Historical loads analysed: {total_loads}",
            f"Unique steel items observed: {unique_items}",
            "",
            "Most frequent loading combinations:",
        ]

        for rank, (combo, count) in enumerate(
            self.history.combo_frequency.most_common(self.top_k_combos), start=1
        ):
            combo_list = sorted(combo)
            summary_lines.append(f"{rank}. {combo_list} -> {count} occurrences")

        if len(summary_lines) == 4:
            summary_lines.append("No repeated combinations found.")

        summary_lines.append("")
        summary_lines.append("Notable pair co-occurrences:")
        for rank, (pair, count) in enumerate(
            self.history.pair_frequency.most_common(self.top_k_combos), start=1
        ):
            pair_list = sorted(pair)
            summary_lines.append(f"{rank}. {pair_list} -> {count} co-loads")

        if len(summary_lines) <= 6:
            summary_lines.append("No frequent pairs detected.")

        # Add physical attribute statistics
        summary_lines.append("")
        summary_lines.append("Physical attribute statistics:")
        attrs = self.history.item_attributes
        if attrs:
            # Calculate statistics for numeric attributes
            numeric_cols = ["LENGTH", "WIDTH", "HEIGHT", "DIAMETER", 
                           "PIECES", "WEIGHT"]
            for col in numeric_cols:
                values = [v.get(col) for v in attrs.values() if v.get(col) is not None]
                if values:
                    try:
                        numeric_vals = [float(v) for v in values]
                        summary_lines.append(
                            f"  {col}: min={min(numeric_vals):.2f}, "
                            f"max={max(numeric_vals):.2f}, "
                            f"avg={sum(numeric_vals)/len(numeric_vals):.2f}"
                        )
                    except (ValueError, TypeError):
                        pass
            
            # Shape distribution
            # shapes = [v.get("SHAPE") for v in attrs.values() if v.get("SHAPE")]
            # if shapes:
            #     shape_counter = Counter(shapes)
            #     summary_lines.append("  Shape distribution:")
            #     for shape, count in shape_counter.most_common(5):
            #         summary_lines.append(f"    {shape}: {count} items")

        return "\n".join(summary_lines)

    def get_context_summary(self) -> str:
        """Public accessor for the textual summary of historical patterns."""
        return self._build_context_summary()
    
    def find_similar_loads(
        self,
        new_items: Sequence[str],
        item_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find similar historical loading cases (LOAD_NO) based on input items.
        
        Args:
            new_items: List of ORDER_NO identifiers
            item_attributes: Optional dict mapping ORDER_NO to physical attributes
            top_k: Number of similar cases to return
            
        Returns:
            List of tuples: (LOAD_NO, similarity_score, load_details)
        """
        new_items_set = set(str(item).strip() for item in new_items)
        
        # Calculate aggregate statistics for new items
        new_total_weight_kg = 0.0
        new_total_weight_ton = 0.0
        new_total_pieces = 0
        
        if item_attributes:
            for item_no in new_items:
                attrs = item_attributes.get(item_no, {})
                if attrs and attrs.get("WEIGHT"):
                    ton_value = float(attrs["WEIGHT"])
                    new_total_weight_ton += ton_value
                    new_total_weight_kg += ton_value * 1000
                    if attrs.get("PIECES"):
                        new_total_pieces += int(attrs["PIECES"])
        
        similarities = []
        
        for load_no, load_info in self.history.load_details.items():
            hist_items_set = load_info["item_set"]
            
            # Calculate Jaccard similarity for item sets
            intersection = len(new_items_set & hist_items_set)
            union = len(new_items_set | hist_items_set)
            jaccard_sim = intersection / union if union > 0 else 0.0
            
            # Calculate weight similarity (if available)
            weight_sim = 0.0
            if new_total_weight_kg > 0 and load_info["total_weight_kg"] > 0:
                weight_diff = abs(new_total_weight_kg - load_info["total_weight_kg"])
                weight_avg = (new_total_weight_kg + load_info["total_weight_kg"]) / 2
                weight_sim = 1.0 - min(weight_diff / weight_avg, 1.0) if weight_avg > 0 else 0.0
            
            # Calculate pieces similarity
            pieces_sim = 0.0
            if new_total_pieces > 0 and load_info["total_pieces"] > 0:
                pieces_diff = abs(new_total_pieces - load_info["total_pieces"])
                pieces_avg = (new_total_pieces + load_info["total_pieces"]) / 2
                pieces_sim = 1.0 - min(pieces_diff / pieces_avg, 1.0) if pieces_avg > 0 else 0.0
            
            # Combined similarity score (weighted)
            # Jaccard similarity is most important (50%), weight and pieces each 25%
            combined_sim = (
                jaccard_sim * 0.5 +
                weight_sim * 0.25 +
                pieces_sim * 0.25
            )
            
            similarities.append((load_no, combined_sim, load_info))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def plan_with_history(self, new_items: Sequence[str]) -> List[List[str]]:
        """
        Generate a baseline loading plan using the most frequent historical combinations.
        """
        remaining = Counter(str(item).strip() for item in new_items if str(item).strip())
        plan: List[List[str]] = []

        # Sort combos by descending size and frequency to prioritise larger, popular templates.
        sorted_combos = sorted(
            self.history.combo_frequency.items(),
            key=lambda kv: (-len(kv[0]), -kv[1]),
        )

        for combo, _ in sorted_combos:
            combo_list = list(combo)
            if not combo_list:
                continue
            if all(remaining[item] > 0 for item in combo_list):
                assigned = []
                for item in combo_list:
                    if remaining[item] > 0:
                        remaining[item] -= 1
                        assigned.append(item)
                if assigned:
                    plan.append(assigned)

        # Assign any leftover items individually.
        for item, count in list(remaining.items()):
            for _ in range(count):
                plan.append([item])

        return plan

    def plan_with_openai(
        self,
        new_items: Sequence[str],
        item_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
        temperature: float = 0.1,
        use_history_plan: bool = True,
        weight_min:int = 10,
        combine_max:int = 3,
    ) -> Dict[str, Any]:
        """
        Ask an OpenAI model to propose a loading plan for the new items.

        Args:
            new_items: List of ORDER_NO identifiers
            item_attributes: Optional dict mapping ORDER_NO to physical attributes
                (length, width, height, diameter, weight, pieces, shape, etc.)
            temperature: Generation temperature
            use_history_plan: Whether to include baseline plan

        Returns a dictionary with keys:
        - "baseline_plan": plan generated from history (optional)
        - "response_text": natural-language recommendation from the model
        """
        if not self.client:
            raise RuntimeError(
                "OpenAI client is not initialised. Set the OPENAI_API_KEY environment variable "
                "or pass the api key explicitly when constructing SteelLoadingPlanner."
            )

        baseline_plan: Optional[List[List[str]]] = None
        if use_history_plan:
            baseline_plan = self.plan_with_history(new_items)

        new_items_list = [str(item).strip() for item in new_items if str(item).strip()]
        
        # Find similar historical loading cases
        similar_loads = self.find_similar_loads(new_items, item_attributes, top_k=5)
        
        system_prompt = (
            "You are a logistics planner specialising in loading plans for steel transport. "
            "Consider vehicle capacity and total weight of orders for each projectID when making loading decisions. "
            "Answer in concise natural language, clearly listing each truck and its assigned steel orders."
        )

        # Build prompt with similar cases
        user_prompt = f"New steel orders to load (ORDER_NO): {new_items_list}\n\n"
        
        # if similar_loads:
        #     user_prompt += "Similar historical loading cases found:\n\n"
        #     for idx, (load_no, sim_score, load_info) in enumerate(similar_loads, 1):
        #         user_prompt += f"Case {idx} (LOAD_NO: {load_no}, Similarity: {sim_score:.2%}):\n"
        #         user_prompt += f"  Items: {', '.join(load_info['items'])}\n"
        #         user_prompt += f"  Total weight: {load_info['total_weight_kg']:.2f} kg"
        #         if load_info['total_weight_ton'] > 0:
        #             user_prompt += f" ({load_info['total_weight_ton']:.3f} ton)"
        #         user_prompt += f"\n  Total pieces: {load_info['total_pieces']}\n"
                
        #         # Add item details with attributes
        #         if load_info['item_details']:
        #             user_prompt += "  Item details:\n"
        #             for item_detail in load_info['item_details'][:10]:  # Limit to first 10 items
        #                 item_no = item_detail['ORDER_NO']
        #                 attrs = item_detail['attributes']
        #                 user_prompt += f"    {item_no}: "
        #                 attr_parts = []
        #                 if attrs.get("WEIGHT"):
        #                     ton_val = attrs["WEIGHT"]
        #                     attr_parts.append(f"Weight={ton_val:.3f} ton")
        #                     attr_parts.append(f"Weight={ton_val * 1000:.1f} kg")
        #                 if attrs.get("PIECES"):
        #                     attr_parts.append(f"Pieces={int(attrs['PIECES'])}")
        #                 if attrs.get("CUSTOMERID"):
        #                     attr_parts.append(f"CustomerID={attrs['CUSTOMERID']}")
        #                 if attrs.get("PROJECTID"):
        #                     attr_parts.append(f"ProjectID={attrs['PROJECTID']}")
        #                 if attrs.get("POST_SECTOR"):
        #                     attr_parts.append(f"PostalSector={attrs['POST_SECTOR']}")
        #                 user_prompt += ", ".join(attr_parts) + "\n"
        #         user_prompt += "\n"
        # else:
        #     # Fallback to general summary if no similar cases found
        #     context_summary = self._build_context_summary()
        #     user_prompt += f"Historical summary:\n{context_summary}\n\n"
      

        # Add physical attributes for new items with better formatting
        def format_attribute_value(key: str, value: Any) -> str:
            """Format attribute value for display."""
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                if key == "WEIGHT":
                    return f"{value:.3f} ton ({value * 1000:.1f} kg)"
                elif key == "PIECES":
                    return f"{int(value)} pieces"
                elif key == "POSTAL_SECTOR":
                    return f"{int(value)}"
                else:
                    return str(value)
            return str(value)
        
        def format_attributes_for_display(attrs: Dict[str, Any], source: str = "") -> List[str]:
            """Format attributes into readable lines."""
            attr_lines = []
            # Map column names to display names for better readability
            attr_mapping = {
                "PROJECTID": "ProjectID",
                "CUSTOMERID": "CustomerID",
                "VEHICLE_TYPE": "VehicleType",
                "POSTAL_SECTOR": "PostalSector",
                "WEIGHT": "Weight",
                "PIECES": "Pieces",
                "REQ_DATE": "RequireDate",
                "LOAD_NO": "LoadNo"
            }
            
            for key, value in attrs.items():
                if value is not None:
                    display_key = attr_mapping.get(key, key)
                    formatted_value = format_attribute_value(key, value)
                    attr_lines.append(f"    {display_key}: {formatted_value}")
            return attr_lines
        
        if item_attributes:
            user_prompt += "\nPhysical attributes of new orders:\n"
            for item_no in new_items_list:
                attrs = item_attributes.get(item_no, {})
                if attrs and any(v is not None for v in attrs.values()):
                    attr_lines = [f"  {item_no}:"]
                    attr_lines.extend(format_attributes_for_display(attrs))
                    user_prompt += "\n".join(attr_lines) + "\n"
                else:
                    # Try to get from historical data if available
                    hist_attrs = self.history.item_attributes.get(item_no, {})
                    if hist_attrs and any(v is not None for v in hist_attrs.values()):
                        attr_lines = [f"  {item_no} (from history):"]
                        attr_lines.extend(format_attributes_for_display(hist_attrs, "history"))
                        user_prompt += "\n".join(attr_lines) + "\n"
        else:
            # Try to get attributes from historical data
            user_prompt += "\nPhysical attributes (from historical data if available):\n"
            for item_no in new_items_list:
                attrs = self.history.item_attributes.get(item_no, {})
                if attrs and any(v is not None for v in attrs.values()):
                    attr_lines = [f"  {item_no}:"]
                    attr_lines.extend(format_attributes_for_display(attrs))
                    user_prompt += "\n".join(attr_lines) + "\n"

        # if baseline_plan:
        #     user_prompt += (
        #         "\nBaseline plan from historical templates:\n"
        #         f"{baseline_plan}\n"
        #         "You may refine this plan if needed. "
        #     )

        user_prompt += (
            "\nPlease provide a loading plan considering:\n"
            "- All orders shall be loaded to assigned vehicle type.\n"
            "- Orders of each projectID shall be loaded to one vehicle where possible.\n"
            "- if the total weight of orders for each projectID is above " + str(weight_min) + " Ton, no need to combine with other projectID.\n"
            "- If total weight of projectID is below " + str(weight_min) + " Ton, it shall be combined with other projectID, and the maximum number of combined projectID is " + str(combine_max) + ".\n"
            #"- Calculate distance between the projects that to be loaded to one vehicle using POSTAL_SECTOR in Singapore to ensure the distance between project location not exceeds 8 KM limit for mixed projectID.\n"
            "- Ensure the loading is not exceeded the maximum weight of 24 Ton for vehicle type TR40/24, 30 Ton for vehicle type LB30, 10 Ton for vehicle type HC.\n"
            "- List each truck load explicitly with projectID and order number."
            #"- Physical dimensions and weight constraints"
            #"- Historical loading patterns"
            #"- Efficient space utilization"
        )

        response_text = self._call_openai(system_prompt, user_prompt, temperature)

        # Format similar loads for return
        similar_cases = []
        for load_no, sim_score, load_info in similar_loads:
            similar_cases.append({
                "load_no": load_no,
                "similarity": sim_score,
                "items": load_info["items"],
                "total_weight_kg": load_info["total_weight_kg"],
                "total_weight_ton": load_info["total_weight_ton"],
                "total_pieces": load_info["total_pieces"],
            })

        return {
            "baseline_plan": baseline_plan or [],
            "response_text": response_text,
            "similar_cases": similar_cases,
            "user_prompt":user_prompt,
        }

    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                #temperature=temperature,
            )
            return response.output[0].content[0].text.strip()  # type: ignore[attr-defined]
        except AttributeError:
            # Fall back to chat completions for older client versions.
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                #temperature=temperature,
            )
            return completion.choices[0].message.content.strip()  # type: ignore[attr-defined]

def _normalize_item_input(raw_input: str) -> List[str]:
    """Parse user input that may contain commas, whitespace, or new lines."""
    import re

    items: List[str] = []
    for line in raw_input.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [token.strip() for token in re.split(r"[,\s]+", line) if token.strip()]
        items.extend(parts)
    return items


def _plan_to_dataframe(plan: Sequence[Sequence[str]]) -> pd.DataFrame:
    if not plan:
        return pd.DataFrame(columns=["Load #", "Items"])
    return pd.DataFrame(
        {
            "Load #": [idx + 1 for idx, _ in enumerate(plan)],
            "Items": [", ".join(load) for load in plan],
        }
    )


def _load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load uploaded file (xlsx or csv) and return as DataFrame.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame with the file contents
    """
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format. Please upload .xlsx, .xls, or .csv file.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")


def _validate_uploaded_file_columns(df: pd.DataFrame, reference_columns: set) -> Tuple[bool, Optional[str]]:
    """
    Validate that uploaded file has required columns.
    
    Args:
        df: DataFrame from uploaded file
        reference_columns: Set of required column names
        
    Returns:
        (is_valid, error_message)
    """
    if "ORDER_NO" not in df.columns:
        return False, "The uploaded file must contain an 'ORDER_NO' column."
    
    # Check if all reference columns exist (ORDER_NO is required, others are optional)
    missing_required = {"ORDER_NO"} - set(df.columns)
    if missing_required:
        return False, f"Missing required columns: {missing_required}"
    
    return True, None


def _extract_items_from_uploaded_file(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Extract ORDER_NO list and attributes from uploaded file.
    For duplicate ORDER_NO, aggregate pieces and weights.
    
    Args:
        df: DataFrame from uploaded file
        
    Returns:
        (item_list, item_attributes_dict)
    """
    # Ensure ORDER_NO is string
    df["ORDER_NO"] = df["ORDER_NO"].astype(str).str.strip()
    
    # Extract attributes for each item
    attribute_columns = [
        "PROJECTID", "CUSTOMERID", "VEHICLE_TYPE", "POSTAL_SECTOR", "WEIGHT",
        "PIECES", "REQ_DATE", "LOAD_NO"
    ]
    
    item_attributes: Dict[str, Dict[str, Any]] = {}
    items: List[str] = []
    
    # Group by ORDER_NO to handle duplicates
    grouped = df.groupby("ORDER_NO")
    
    for item_no, item_group in grouped:
        items.append(item_no)
        attrs: Dict[str, Any] = {}
        
        # For dimensions and shape, use first non-null value
        for col in ["PROJECTID","CUSTOMERID", "VEHICLE_TYPE", "POSTAL_SECTOR","REQ_DATE", "LOAD_NO"]:
            if col in df.columns:
                # Get first non-null value
                values = item_group[col].dropna()
                if len(values) > 0:
                    value = values.iloc[0]
                    if col == "CUSTOMERID":
                        attrs[col] = str(value) if pd.notna(value) else None
                    elif col == "PROJECTID":
                        attrs[col] = str(value) if pd.notna(value) else None
                    elif col == "POSTAL_SECTOR":
                        attrs[col] = str(value) if pd.notna(value) else None
                    elif col == "VEHICLE_TYPE":
                        attrs[col] = str(value) if pd.notna(value) else None
                    elif col == "REQ_DATE":
                        attrs[col] = str(value) if pd.notna(value) else None
                    elif col == "LOAD_NO":
                        attrs[col] = str(value) if pd.notna(value) else None
                    else:
                        try:
                            attrs[col] = float(value)
                        except (ValueError, TypeError):
                            attrs[col] = value
                else:
                    attrs[col] = None
            else:
                attrs[col] = None
        
        # For pieces and weights, sum across all rows for the same ORDER_NO
        for col in ["PIECES", "WEIGHT"]:
            if col in df.columns:
                values = item_group[col].dropna()
                if len(values) > 0:
                    try:
                        numeric_values = [float(v) for v in values]
                        total_value = sum(numeric_values)
                        attrs[col] = total_value
                    except (ValueError, TypeError):
                        # If conversion fails, use first value
                        attrs[col] = values.iloc[0]
                else:
                    attrs[col] = None
            else:
                attrs[col] = None
        
        item_attributes[item_no] = attrs
    
    return items, item_attributes


def _parse_loading_plan_from_response(response_text: str, known_items: Optional[List[str]] = None) -> List[List[str]]:
    """
    Parse loading plan from OpenAI response text.
    Attempts to extract truck assignments from natural language.
    
    Args:
        response_text: Natural language response from OpenAI
        known_items: Optional list of known ORDER_NOs to filter results
        
    Returns:
        List of lists, where each inner list contains ORDER_NOs assigned to one truck
    """
    import re
    
    plan: List[List[str]] = []
    known_items_set = set(item.upper() for item in known_items) if known_items else None
    
    # Try to find patterns like "Truck 1: ITEM1, ITEM2" or "Load 1: [ITEM1, ITEM2]"
    # Look for numbered trucks/loads
    patterns = [
        r'(?:Truck|Load|Vehicle|车|车辆)\s*(\d+)[:：]\s*([^\n]+)',
        r'(?:Truck|Load|Vehicle|车|车辆)\s*(\d+)[:：]\s*\[([^\]]+)\]',
        r'(\d+)\.\s*(?:Truck|Load|Vehicle|车|车辆)[:：]\s*([^\n]+)',
        r'第\s*(\d+)\s*(?:辆|车|次)[:：]\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            for match in matches:
                items_str = match[1] if len(match) > 1 else match[0]
                # Extract ORDER_NOs (alphanumeric codes, more flexible pattern)
                items = re.findall(r'\b[A-Z0-9_]{3,}\b', items_str.upper())
                # Filter to known items if provided
                if known_items_set:
                    items = [item for item in items if item in known_items_set]
                if items:
                    plan.append(items)
            if plan:
                break
    
    # If no structured pattern found, try to extract by lines
    if not plan:
        lines = response_text.split('\n')
        current_load = []
        for line in lines:
            # Check if line indicates a new load
            if re.search(r'(?:Truck|Load|Vehicle|车|车辆|第.*[辆车次])', line, re.IGNORECASE):
                if current_load:
                    plan.append(current_load)
                current_load = []
            
            # Extract items from line
            line_items = re.findall(r'\b[A-Z0-9_]{3,}\b', line.upper())
            if known_items_set:
                line_items = [item for item in line_items if item in known_items_set]
            if line_items:
                current_load.extend(line_items)
        
        if current_load:
            plan.append(current_load)
    
    # If still no plan, try to extract all ORDER_NOs and group them
    if not plan and known_items:
        all_items = []
        for item in known_items:
            if item.upper() in response_text.upper():
                all_items.append(item)
        
        if all_items:
            # Try to group by lines or paragraphs
            paragraphs = response_text.split('\n\n')
            for para in paragraphs:
                para_items = [item for item in all_items if item.upper() in para.upper()]
                if para_items:
                    plan.append(para_items)
            
            # If still no grouping, put all in one load
            if not plan and all_items:
                plan.append(all_items)
    
    return plan


def _plan_to_output_dataframe(
    plan: List[List[str]],
    item_attributes: Dict[str, Dict[str, Any]],
    reference_columns: List[str],
    load_no_prefix: str = "LOAD"
) -> pd.DataFrame:
    """
    Convert loading plan to DataFrame with same structure as input file.
    
    Args:
        plan: List of lists, each inner list is items in one truck/load
        item_attributes: Dictionary mapping ORDER_NO to attributes
        reference_columns: Column names from reference file
        load_no_prefix: Prefix for LOAD_NO generation
        
    Returns:
        DataFrame with same columns as input file
    """
    rows = []
    
    for load_idx, load_items in enumerate(plan, start=1):
        load_no = f"{load_no_prefix}_{load_idx:04d}"
        
        for item_no in load_items:
            # Get attributes for this item
            attrs = item_attributes.get(item_no, {})
            
            # Create row with all reference columns
            row = {}
            
            # Set LOAD_NO
            row["LOAD_NO"] = load_no
            
            # Set ORDER_NO
            row["ORDER_NO"] = item_no
            
            # Fill in physical attributes
            for col in reference_columns:
                if col not in ["LOAD_NO", "ORDER_NO"]:
                    if col in attrs and attrs[col] is not None:
                        row[col] = attrs[col]
                    else:
                        row[col] = None
            
            rows.append(row)
    
    if not rows:
        # Return empty DataFrame with reference columns
        return pd.DataFrame(columns=reference_columns)
    
    df = pd.DataFrame(rows)
    
    # Ensure all reference columns are present
    for col in reference_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns to match reference
    df = df[reference_columns]
    
    return df


def run_streamlit_app() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Streamlit is required to run the interactive interface. "
            "Install it with `pip install streamlit`."
        ) from exc

    st.set_page_config(page_title="Steel Loading Planner", layout="wide")
    st.title("Steel Loading Planner")
    st.write(
        "Use historical loading records and OpenAI-assisted reasoning to craft loading plans "
        "for new steel shipments."
    )

    with st.sidebar:
        st.header("Configuration")

        default_key = os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_KEY)
        openai_key_input = st.text_input(
            "OpenAI API key",
            value=default_key,
            type="password",
            help="The key is used client-side only for this session.",
        )

        # Advanced settings (hidden by default)
        base_data_path = "data.xlsx"
        base_model_name = "gpt-4.1-mini"
        with st.expander("Advanced model & data settings", expanded=False):
            st.caption("Adjust only if you need to override default data/model paths.")
            data_path_input = st.text_input("Historical data file", base_data_path, key="data_path_input")
            model_name_input = st.text_input("OpenAI model", base_model_name, key="model_name_input")

        resolved_data_path = data_path_input.strip() or "data.xlsx"

        temperature = st.slider(
            "Generation temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05
        )
        weight_min = st.slider(
            "Min Weight per Load", min_value=5, max_value=20, value=10, step=1
        )
        combine_max = st.slider(
            "Max Combined Projects", min_value=1, max_value=10, value=3, step=1
        )
        use_history_plan = st.checkbox(
            "Provide historical baseline to OpenAI", value=True,
            help="If enabled, the baseline plan generated from historical patterns will be passed "
                 "to the model as a starting point.",
        )
        top_k = st.slider(
            "Number of frequent combinations to summarise",
            min_value=3,
            max_value=200,
            value=10,
        )
        show_history = st.checkbox("Show historical summary", value=True)

    planner: Optional[SteelLoadingPlanner] = None
    planner_error: Optional[str] = None
    data_file = Path(resolved_data_path)

    if not data_file.exists():
        planner_error = f"Data file '{data_file}' not found."
    else:
        try:
            planner = SteelLoadingPlanner(
                data_path=str(data_file),
                openai_api_key=openai_key_input.strip() or None,
                model=model_name_input.strip() or "gpt-4.1-mini",
                top_k_combos=top_k,
            )
        except Exception as exc:  # pylint: disable=broad-except
            planner_error = str(exc)

    st.subheader("New Load Request")
    st.write(
        "Upload a table file (xlsx or csv) containing steel items to load. "
        "The file should have the same column structure as the historical data file (data.xlsx), "
        "with at least an 'ORDER_NO' column. Physical attributes will be extracted from the file if available."
    )

    uploaded_file = st.file_uploader(
        "Upload steel items file",
        type=["xlsx", "xls", "csv"],
        help="Upload a file with columns matching data.xlsx structure (ORDER_NO, LENGTH, WIDTH, HEIGHT, etc.)"
    )
    
    parsed_items: List[str] = []
    item_attributes: Optional[Dict[str, Dict[str, Any]]] = None
    uploaded_df: Optional[pd.DataFrame] = None
    
    if uploaded_file is not None:
        try:
            # Load the uploaded file
            uploaded_df = _load_uploaded_file(uploaded_file)
            
            # Validate columns
            if planner:
                reference_columns = set(planner.dataframe.columns)
            else:
                reference_columns = {"ORDER_NO"}
            
            is_valid, error_msg = _validate_uploaded_file_columns(uploaded_df, reference_columns)
            
            if not is_valid:
                st.error(error_msg)
            else:
                # Extract items and attributes
                parsed_items, item_attributes = _extract_items_from_uploaded_file(uploaded_df)
                
                st.success(f"Successfully loaded {len(parsed_items)} unique steel items from the file.")
                
                # Display preview
                with st.expander("Preview uploaded data", expanded=False):
                    st.write(f"**Total rows:** {len(uploaded_df)}")
                    st.write(f"**Unique items:** {len(parsed_items)}")
                    st.dataframe(uploaded_df.head(10), width="stretch")
                
                # Display extracted items and attributes
                if parsed_items:
                    st.write(f"**Items to load:** {len(parsed_items)} unique items")
                    
                    # Display detailed attributes table
                    if item_attributes:
                        attrs_with_data = {
                            item: attrs for item, attrs in item_attributes.items()
                            if any(v is not None for v in attrs.values())
                        }
                        if attrs_with_data:
                            st.info(f"Physical attributes found for {len(attrs_with_data)} items.")
                            
                            # Create a summary table of attributes
                            with st.expander("View extracted physical attributes", expanded=True):
                                attr_rows = []
                                for item_no, attrs in item_attributes.items():
                                    row = {"ORDER_NO": item_no}
                                    # Add physical attributes
                                    row["ProjectID"] = attrs.get("PROJECTID", "N/A")
                                    row["CustomerID"] = attrs.get("CUSTOMERID", "N/A")
                                    row["VehicleType"] = attrs.get("VEHICLE_TYPE", "N/A")
                                    row["PostalSector"] = attrs.get("POSTAL_SECTOR", "N/A")
                                    row["RequireDate"] = attrs.get("REQ_DATE", "N/A")
                                    row["LoadNo"] = attrs.get("LOAD_NO", "N/A")
                                    ton_value = attrs.get("WEIGHT")
                                    if isinstance(ton_value, (int, float)):
                                        row["Weight (kg)"] = f"{ton_value * 1000:.1f}"
                                        row["Weight (ton)"] = f"{ton_value:.3f}"
                                    else:
                                        row["Weight (kg)"] = "N/A"
                                        row["Weight (ton)"] = ton_value if ton_value is not None else "N/A"
                                    row["Pieces"] = attrs.get("PIECES", "N/A")
                                    attr_rows.append(row)
                                
                                if attr_rows:
                                    attr_df = pd.DataFrame(attr_rows)
                                    st.dataframe(attr_df, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            uploaded_df = None

    openai_result: Optional[Dict[str, Any]] = None
    
    if st.button("Generate plan with OpenAI", type="primary"):
        if planner is None:
            st.error(planner_error or "Planner could not be initialised.")
        elif not parsed_items:
            st.warning("Please upload a file with steel items first.")
        else:
            try:
                openai_result = planner.plan_with_openai(
                    parsed_items,
                    item_attributes=item_attributes,
                    temperature=temperature,
                    use_history_plan=use_history_plan,
                    weight_min=weight_min,
                    combine_max=combine_max,
                )
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to retrieve response from OpenAI: {exc}")

    if openai_result is not None:
        st.subheader("Plan suggested by OpenAI")
        response_text = openai_result.get("response_text", "")
        if response_text:
            st.markdown(response_text)
        else:
            st.warning("The model did not return any content.")

        # Display user_prompt
        user_prompt = openai_result.get("user_prompt", [])
        if user_prompt:
            with st.expander("User prompting", expanded=False):
                st.write(user_prompt)

        # Display similar historical cases used
        similar_cases = openai_result.get("similar_cases", [])
        if similar_cases:
            with st.expander("Similar historical loading cases used as reference", expanded=False):
                for idx, case in enumerate(similar_cases, 1):
                    st.write(f"**Case {idx}** (LOAD_NO: {case['load_no']}, Similarity: {case['similarity']:.1%})")
                    items_display = ', '.join(case['items'][:10])
                    if len(case['items']) > 10:
                        items_display += f" ... and {len(case['items']) - 10} more items"
                    st.write(f"- Items: {items_display}")
                    weight_text = f"- Total weight: {case['total_weight_kg']:.2f} kg"
                    if case['total_weight_ton'] > 0:
                        weight_text += f" ({case['total_weight_ton']:.3f} ton)"
                    st.write(weight_text)
                    st.write(f"- Total pieces: {case['total_pieces']}")
                    if idx < len(similar_cases):
                        st.divider()

        # Display physical attributes used in planning
        if item_attributes and parsed_items:
            with st.expander("Physical attributes used for planning", expanded=False):
                attr_summary_rows = []
                for item_no in parsed_items:
                    attrs = item_attributes.get(item_no, {})
                    row = {"ORDER_NO": item_no}
                    # Format attributes for display
                    row["ProjectID"] = attrs.get("PROJECTID", "N/A")
                    row["CustomerID"] = attrs.get("CUSTOMERID", "N/A")
                    row["VehicleType"] = attrs.get("VEHICLE_TYPE", "N/A")
                    row["PostalSector"] = attrs.get("POSTAL_SECTOR", "N/A")
                    row["RequireDate"] = attrs.get("REQ_DATE", "N/A")
                    row["LoadNo"] = attrs.get("LOAD_NO", "N/A")
                    if attrs.get("WEIGHT") is not None and isinstance(attrs['WEIGHT'], (int, float)):
                        row["Weight (ton)"] = f"{attrs['WEIGHT']:.3f}"
                        row["Weight (kg)"] = f"{attrs['WEIGHT'] * 1000:.1f}"
                    else:
                        row["Weight (ton)"] = attrs.get("WEIGHT", "N/A")
                        row["Weight (kg)"] = "N/A"
                    if attrs.get("PIECES") is not None:
                        row["Pieces"] = int(attrs['PIECES']) if isinstance(attrs['PIECES'], (int, float)) else attrs['PIECES']
                    else:
                        row["Pieces"] = "N/A"
                    attr_summary_rows.append(row)
                
                if attr_summary_rows:
                    attr_summary_df = pd.DataFrame(attr_summary_rows)
                    st.dataframe(attr_summary_df, width="stretch", hide_index=True)

        if openai_result.get("baseline_plan"):
            with st.expander("Baseline provided to the model"):
                st.dataframe(
                    _plan_to_dataframe(openai_result["baseline_plan"]),
                    width="stretch",
                )

        # Generate output CSV with same format as input file
        output_df: Optional[pd.DataFrame] = None
        if response_text and item_attributes and parsed_items:
            try:
                # Parse loading plan from response
                parsed_plan = _parse_loading_plan_from_response(response_text, known_items=parsed_items)
                
                # If parsing failed, try using baseline plan
                if not parsed_plan and openai_result.get("baseline_plan"):
                    parsed_plan = openai_result["baseline_plan"]
                
                if parsed_plan:
                    # Get reference columns from uploaded file or planner
                    if uploaded_df is not None:
                        reference_columns = uploaded_df.columns.tolist()
                    elif planner:
                        reference_columns = planner.dataframe.columns.tolist()
                    else:
                        # Default columns if neither available
                        reference_columns = ["LOAD_NO", "ORDER_NO", "PROJECTID","CUSTOMERID", "VEHICLE_TYPE", "POSTAL_SECTOR","WEIGHT",
                                             "PIECES", "REQ_DATE", "LOAD_NO"]
                    
                    # Convert plan to DataFrame
                    output_df = _plan_to_output_dataframe(
                        parsed_plan,
                        item_attributes,
                        reference_columns
                    )
                    
                    # Display preview
                    with st.expander("Preview output table (same format as input)", expanded=False):
                        st.write(f"**Total rows:** {len(output_df)}")
                        st.write(f"**Number of loads:** {output_df['LOAD_NO'].nunique() if 'LOAD_NO' in output_df.columns else 0}")
                        st.dataframe(output_df.head(20), width="stretch")
            except Exception as e:
                st.warning(f"Could not generate output table: {str(e)}")
                output_df = None

        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download button
            if output_df is not None and len(output_df) > 0:
                csv_data = output_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "Download as CSV",
                    data=csv_data,
                    file_name="loading_plan_output.csv",
                    mime="text/csv",
                    help="Download the loading plan in CSV format matching the input file structure"
                )
            else:
                st.download_button(
                    "Download as CSV",
                    data="",
                    file_name="loading_plan_output.csv",
                    mime="text/csv",
                    disabled=True,
                    help="No valid plan to export"
                )
        
        with col2:
            # JSON download button
            download_data = openai_result.copy()
            if item_attributes:
                download_data["item_attributes"] = item_attributes
            if output_df is not None:
                download_data["output_table"] = output_df.to_dict('records')
            
            st.download_button(
                "Download as JSON",
                data=json.dumps(download_data, indent=2, ensure_ascii=False, default=str),
                file_name="loading_plan.json",
                mime="application/json",
            )

    if planner_error and planner is None:
        st.error(planner_error)

    if planner and show_history:
        st.subheader("Historical insights")
        st.code(planner.get_context_summary())


def demo() -> None:
    """
    Example entry point.
    Set the environment variable OPENAI_API_KEY before running.
    """
    planner = SteelLoadingPlanner(
        data_path="data.xlsx",
        openai_api_key=os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_KEY,
    )

    # Example: replace with actual incoming steel item IDs.
    new_items = ["ITEM_101", "ITEM_202", "ITEM_303", "ITEM_404"]

    print("Baseline plan based on historical data:")
    print(planner.plan_with_history(new_items))

    if planner.client:
        print("\nPlan generated with OpenAI:")
        result = planner.plan_with_openai(new_items)
        print(result.get("response_text", "No response received."))
    else:
        print("\nOpenAI API key not provided. Skipping model-generated plan.")


def running_in_streamlit() -> bool:
    """Best-effort check to see if the script is being executed inside Streamlit."""
    return any(module_name.startswith("streamlit") for module_name in sys.modules)


if __name__ == "__main__":
    if running_in_streamlit():
        run_streamlit_app()
    else:
        demo()

